import ollama
import numpy as np
from heapq import heappop, heappush

def merge_most_similar_adjacent(chunks, embeddings, num_merges):
    n = len(chunks)
    if n <= 1 or num_merges <= 0:
        return chunks, embeddings
    
    # Initialize heap with all adjacent pairs
    heap = []
    for i in range(n - 1):
        similarity = np.dot(embeddings[i], embeddings[i+1])
        heappush(heap, (-similarity, i, i+1))  # Store (i, j) for validation
    # Track valid indices (all start as valid)
    valid = set(range(n))
    
    merge_count = 0
    while merge_count < num_merges and len(valid) > 1 and heap:
        # Get the most similar valid pair
        neg_sim, i, j = heappop(heap)
        if i not in valid or j not in valid:
            continue  # Skip invalid pairs

        # Merge chunks and embeddings
        merged_chunk = chunks[i] + ". " + chunks[j]
        merged_embedding = embeddings[i] + embeddings[j]
        merged_embedding /= np.linalg.norm(merged_embedding)
        
        # Update chunks and embeddings
        chunks[i] = merged_chunk
        embeddings[i] = merged_embedding
        valid.remove(j)  # Mark j as merged
        
        # Invalidate all heap entries involving i or j
        # (We'll push new entries for i's neighbors later)
        
        # Add new similarities for (i-1, i) and (i, new j's right neighbor)
        if i - 1 in valid:
            new_sim = np.dot(embeddings[i-1], embeddings[i])
            heappush(heap, (-new_sim, i-1, i))
        new_j = j + 1
        while new_j < n and new_j not in valid:
            new_j += 1  # Find next valid right neighbor
        if new_j < n:
            new_sim = np.dot(embeddings[i], embeddings[new_j])
            heappush(heap, (-new_sim, i, new_j))
        
        merge_count += 1
    
    # Reconstruct the final chunks and embeddings
    sorted_valid = sorted(valid)
    merged_chunks = [chunks[i] for i in sorted_valid]
    merged_embeddings = np.array([embeddings[i] for i in sorted_valid])
    return merged_chunks, merged_embeddings

def semanticChunk(self, text, window=8):
    sentences = list(filter(None, text.strip().split(".")))
    # I know this sentence splitting is ass, I don't really care
    newsentences = []
    tmp = []
    for i in range(len(sentences)):
        tmp.append(sentences[i])
        if len(sentences[i]) > 12:
            if len(sentences) > i+1:
                if len(sentences[i+1]) > 12:
                    newsentences.append(".".join(tmp).strip())
                    tmp = []
            else:
                newsentences.append(".".join(tmp).strip())
                tmp = []
    sentences = newsentences
    newsentences = []
    for i in range(len(sentences)):
        newsentences.append(". ".join(sentences[max(i-window,0):min(i+window+1,len(sentences))]))
    embd = np.array(ollama.embed(model='nomic-embed-text:latest',input=newsentences).embeddings,dtype=np.float16)
    # weights = np.matmul(embd,np.transpose(embd))
    # plt.imshow(weights, cmap='jet', interpolation='nearest')
    # plt.show()
    Chunks = merge_most_similar_adjacent(list(sentences),embd,(len(sentences)*7)//8)[0]
    return Chunks