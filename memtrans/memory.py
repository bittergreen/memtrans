import os
import faiss
import numpy as np
import torch


from hyperparameters import mem_size, n_heads, head_size, topk, device


class MemoryForOneHead:

    def __init__(self, head_id, mem_size=mem_size, head_size=head_size) -> None:
        """
        There are 2 components of this transformer memory:
        - A vector db storing the key embeddings, supports similarity search
        - A memmap storing the values

        Since the vector db(faiss) does not support multiple columns in the way that relational dbs do,
        we kinda have to store keys and values in different places.

        The keys are stored in faiss, supporting later retrieval using new query vectors.
        The values are stored in the memmap, we make sure that the stored kv pairs all share the same ids,
        so that after we retrieve matched key vectors and their ids from faiss, we can then use those ids
        to get the value vectors that's originally paired with those keys.

        Ids are tracked with the self.total_offset
        """
        self.id = head_id
        self.head_size = head_size
        # Create a FAISS index for each key embedding
        self.key_db = faiss.IndexFlatL2(head_size)
        
        # Todo: Change this storage location/remove storage after usage!
        os.makedirs("./mapfiles", exist_ok=True)
        value_db_filepath = f"./mapfiles/memory_head_{self.id}.memmap"
        self.value_db = np.memmap(value_db_filepath, mode='w+', dtype=np.float32, shape=(mem_size, head_size))
        # Todo: Make this atomic!
        self.total_offset = 0

    def is_empty(self):
        return self.total_offset == 0

    def store(self, key, value):
        """
        Originally:
        key: (batch, sequence_length, head_size)
        value: (batch, sequence_length, head_size)
        Store them as shape (batch * sequence_length, head_size)
        """
        b, t = key.shape[0], key.shape[1]
        # Flatten batch and sequence dimensions for FAISS storage: (batch * sequence_length, head_size)
        key_flat = key.reshape(-1, self.head_size)
        # Add the flattened keys to FAISS
        self.key_db.add(key_flat)

        offset = b * t
        ids = torch.arange(self.total_offset, self.total_offset + offset)

        # Add the corresponding values to the memmap. The value is flattened in the same way as the key.
        value_flat = value.reshape(-1, self.head_size)
        self.value_db[ids] = value_flat
        self.value_db.flush()
        self.total_offset += offset

        return ids

    def retrieve(self, query, topk=topk):
        """
        query: (batch, sequence, head_size)
        """
        b, t = query.shape[0], query.shape[1]
        query_flat = query.reshape(-1, self.head_size)

        # matched_keys & matched_values: (batch * sequence_length, topk, embed_dim)
        _, ids, matched_keys = self.key_db.search_and_reconstruct(query_flat, topk)
        matched_values = self.value_db[ids]  # ndarray

        # transpose to (batch * sequence_length, embed_dim, topk)
        matched_keys = np.transpose(matched_keys, (0, 2, 1))
        matched_values = np.transpose(matched_values, (0, 2, 1))

        matched_keys = torch.from_numpy(matched_keys).unflatten(0, (b, t)).to(device=device)
        matched_values = torch.from_numpy(matched_values).unflatten(0, (b, t)).to(device=device)

        # returning (b, t, embed_dim, topk)
        return matched_keys, matched_values

    def clear(self):
        self.key_db.reset()
        self.value_db.flush()
        self.total_offset = 0


class Memory:

    def __init__(self, n_heads=n_heads, mem_size=mem_size, head_size=head_size):
        self.n = n_heads
        self.mem_size = mem_size
        self.head_size = head_size
        self.memory_list = [MemoryForOneHead(i, self.mem_size, self.head_size) for i in range(self.n)]

    def is_empty(self):
        return all(m.is_empty() for m in self.memory_list)

    def store(self, key, value):
        """
        Originally:
        key: (batch, sequence_length, n_heads * head_size)
        value: (batch, sequence_length, n_heads * head_size)
        """
        # Reshape to (batch, sequence_length, n_heads, head_size)
        key = key.unflatten(-1, (self.n, self.head_size)).cpu().detach().numpy()
        value = value.unflatten(-1, (self.n, self.head_size)).cpu().detach().numpy()
        
        # Store in each memory head - transpose to get (n_heads, batch, sequence_length, head_size)
        key_heads = np.transpose(key, (2, 0, 1, 3))
        value_heads = np.transpose(value, (2, 0, 1, 3))
        
        # Store each head's data and collect the ids (n_heads, batch * sequence_length)
        ids = [self.memory_list[i].store(key_heads[i], value_heads[i]) for i in range(self.n)]
        return ids

    def retrieve(self, query, topk=topk):
        """
        query: (batch, sequence_length, n_heads * head_size)
        returns: matched_keys, matched_values both (batch, sequence_length, n_heads * head_size, top_k)
        """
        query = query.unflatten(-1, (self.n, self.head_size)).cpu().detach().numpy()
        query_heads = np.transpose(query, (2, 0, 1, 3))
        
        all_matched_keys = []
        all_matched_values = []
        
        for i in range(self.n):
            matched_keys, matched_values = self.memory_list[i].retrieve(query_heads[i], topk)
            all_matched_keys.append(matched_keys)
            all_matched_values.append(matched_values)
        
        # Stack along head dimension, and then flatten
        # (batch, seq_len, n_heads * head_size, topk)
        matched_keys = torch.stack(all_matched_keys, dim=2).flatten(2, 3)
        matched_values = torch.stack(all_matched_values, dim=2).flatten(2, 3)

        return matched_keys, matched_values

    def clear(self):
        """Clear all memory heads"""
        for m in self.memory_list:
            m.clear()

