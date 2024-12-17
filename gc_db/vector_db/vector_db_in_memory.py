import numpy as np
from sklearn.cluster import KMeans


class VectorDB_IM:

    def __init__(self):
        # {1903:[0.5,0.6,0.7],4:[0.5,0.7....}
        # {4:[1,6,1903],5:[78,89]
        self.inverted_index: dict[int, list[int]] = {}
        self.codebook: list[np.array] = []
        self.db_vector: dict[int, np.array] = {}

    def insert(self, vector: np.array, vector_id: int):
        self.db_vector[vector_id] = vector

    # [(1903,0.40),(75,0.33)...]
    def query(self, query_vector: np.array, k: int = 20) -> list[tuple[int, float]]:
        distances_and_ids = [(vector_id, self.cosine_similarity(query_vector, self.db_vector[vector_id])) for vector_id
                             in self.db_vector.keys()]
        distances_and_ids.sort(key=lambda x: -x[1])
        return distances_and_ids[:k]

    def cosine_similarity(self, vector_a: np.array, vector_b: np.array) -> float:
        num = np.dot(vector_a, vector_b)
        denom = np.dot(np.linalg.norm(vector_a), np.linalg.norm(vector_b))
        return num / denom

    def init_kmeans_index(self, nb_clusters: int = 10):
        kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
        kmeans.fit(list(self.db_vector.values()))

        # [[0.5,0.6,0.7]]
        self.codebook = kmeans.cluster_centers_

        for vector_id in self.db_vector.keys():
            cluster_id = kmeans.predict([self.db_vector[vector_id]])[0]
            if cluster_id in self.inverted_index:
                self.inverted_index[cluster_id].append(vector_id)
            else:
                self.inverted_index[cluster_id] = [vector_id]

    def query_with_kmeans(self, query_vector: np.array, k: int = 20, n_probes: int = 1) -> list[tuple[int, float]]:
        distances_with_centroids = [(centroid_id, self.cosine_similarity(query_vector, centroid)) for
                                    centroid_id, centroid in enumerate(self.codebook)]
        distances_with_centroids.sort(key=lambda x: -x[1])
        nprobes_centroids_ids = distances_with_centroids[:n_probes]

        distance_and_ids = []
        for centroid_id, centroid in nprobes_centroids_ids:
            for vector_id in self.inverted_index[centroid_id]:
                distance_and_ids.append((vector_id, self.cosine_similarity(query_vector, self.db_vector[vector_id])))

        distance_and_ids.sort(key=lambda x: -x[1])
        return distance_and_ids[:k]


if __name__ == "__main__":
    import logging
    import pickle
    import time
    from gc_db.config import DICT_IDS_EMBEDDINGS_PATH
    from fashion_clip.fashion_clip import FashionCLIP

    from threadpoolctl import threadpool_info

    print("threadpool info", threadpool_info())

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting Main")
    dict_ids_embeddings = pickle.load(open(DICT_IDS_EMBEDDINGS_PATH, "rb"))
    VDB_IM = VectorDB_IM()
    if hasattr(VDB_IM, "insert"):
        _ = [VDB_IM.insert(dict_ids_embeddings[id], id) for id in dict_ids_embeddings.keys()]
        logger.info("Loading vector to memory db : " + str(len(VDB_IM.db_vector)))

    if hasattr(VDB_IM, "query"):
        FCLIP = FashionCLIP('fashion-clip')
        embeded_query = FCLIP.encode_text(["White tee shirt with NASA logo"], 1)[0]
        logger.info("QUERY SHAPE:" + str(embeded_query.shape))
        start = time.time()
        nn = VDB_IM.query(embeded_query)
        end = time.time()
        lasted = np.round(end - start, 3)
        logger.info("Time elapsed with exhaustive search:" + str(lasted))
        logger.info("Results : " + str(nn))

    if hasattr(VDB_IM, 'init_kmeans_index'):
        start = time.time()
        VDB_IM.init_kmeans_index(nb_clusters=5)
        end = time.time()
        lasted = np.round(end - start, 3)
        # 2 secondes sur le sample
        # 11 secondes sur le full
        logger.info("Computed kmeans index in " + str(lasted))
        logger.info("Codebook :" + str(VDB_IM.codebook)[:200])
        logger.info("Inverted index:" + str(VDB_IM.inverted_index)[:200])
        logger.info("Inverted index first cluster length: " + str(len(VDB_IM.inverted_index[0])))

    if hasattr(VDB_IM, 'query_with_kmeans'):
        start = time.time()
        nn = VDB_IM.query_with_kmeans(embeded_query, n_probes=3)
        end = time.time()
        lasted = np.round(end - start, 3)
        logger.info("Time elapsed with IVF search:" + str(lasted))
