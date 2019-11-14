import numpy as np
import collections
from module.distanceFunction import distanceFunction
from more_itertools import distinct_permutations

"""
The foundation of this implementation is based on Basic LSH Indexing mentioned in the following paper:
https://www.cs.princeton.edu/cass/papers/mplsh_vldb07.pdf

# TODO
- Using multi-probe LSH to perform better
- Test whether reducing k hash size to get at least t neighbors is a better way
"""

class LSH:
    def __init__(self, L_layers, k_hashes_per_layer, W_parameter = 0.50):
        """
            Must call build_structure with the dataset after creating current instance
        """
        self.L_layers = L_layers                        # number of hash tables
        self.k_hashes_per_layer = k_hashes_per_layer    # the random projection from F dimension to K dimension
        self.feature_size = 0                           # the length of features for each object
        self.W_parameter = W_parameter                  # Parameter used for scaling
        self.hash_tables = []                           # L layers = L hash tables to be used
        self.images = []                                # a list of image ID
        self.data_matrix = None

    def build_structure(self, dataset: dict):
        """
            - Initialization of the current instance with given dataset.
            - Create L new hash tables and each table has hash value with K length

            Arguments
                - dataset: a dictionary with image id as the key and the latent semantics vector as the value
        """
        self.dataset = dataset
        self.images = list(self.dataset.keys())
        self.data_matrix = np.asarray(list(self.dataset.values()))
        self.feature_size = self.data_matrix.shape[1]
        print("Building structure...")
        for layer in range(self.L_layers):
            print(f"Creating {layer}-layer for current structure")
            new_hash_table = HashTable(self.k_hashes_per_layer, self.feature_size, self.W_parameter)
            for index, vector in enumerate(self.data_matrix):
                new_hash_table[vector] = self.images[index]
            self.hash_tables.append(new_hash_table)

    def get_t_most_similar_images(self, query_image, t):
        """
            To search for k similar images, we first query each hash table with the target image and get
            all the candidates in each matching bucket. However, if the amount of candidates do not satisfy
            the t number, we try to alternate bit in every code by one or more and find the candidates in
            corresponding bucket until the number of candidates match the target number. After getting all
            the candidates, we sort them by distance and print the result.

            !!! Not the best way, may try to reduce the dimensions by decreasing hash size, but current
            method still works

            Arguments:
                - query_image: the target image
                - t: t nearest neighbors
        """
        print(f"Getting {t} most similar images of {query_image} from dataset...")
        match_length = self.k_hashes_per_layer
        query_vector = self.dataset[query_image]
        candidates = self.get_candidates(query_image, t)
        calculate_distance = distanceFunction.createDistance("l2")
        candidates_with_distance = []

        for candidate in candidates:
            candidate_vector = self.dataset[candidate]
            distance = calculate_distance(candidate_vector, query_vector)
            candidates_with_distance.append([candidate, candidate_vector, distance])

        candidates_with_distance.sort(key=lambda x: x[2])

        if len(candidates_with_distance) < t:
            print("Only get %s candidates, can't find other possible candidates." % len(candidates_with_distance))
        else:
            candidates_with_distance = candidates_with_distance[:t]

        for index, row in enumerate(candidates_with_distance):
            print("No.{0}  Image ID: {1}  Distance: {2}".format(index, row[0], row[2]))


    def get_candidates(self, query_image, t):
        """
            A helper to get all the possible candidates
        """
        match_length = self.k_hashes_per_layer
        query_vector = self.dataset[query_image]
        all_possible_candidates = set()
        query_code_for_each_hash = [table.generate_hash(query_vector) for table in self.hash_tables]

        while len(all_possible_candidates) < t and match_length >= 0:
            all_query_code_for_each_table = self.generate_permutation_for_query_code(query_code_for_each_hash, match_length)

            for idx, table in enumerate(self.hash_tables):
                candidates_for_current_table = set()

                for code in all_query_code_for_each_table[idx]:
                    bucket = table.get_bucket_objects(code)
                    candidates_for_current_table.update(bucket)
                all_possible_candidates.update(candidates_for_current_table)

            if query_image in all_possible_candidates:
                all_possible_candidates.remove(query_image)
            match_length -= 1

        return list(all_possible_candidates)

    @staticmethod
    def generate_permutation_for_query_code(query_code_for_each_hash, match_length):
        """
            Change one or more bit in the query code to get permutation for each hash table

            Arguments:
                - query_code_for_each_hash: query code for each different hash tables
                - match_length: total bits should be match
        """
        positions = '0' * match_length + '1' * (len(query_code_for_each_hash[0]) - match_length)
        all_permutations = []

        for query_code in query_code_for_each_hash:
            permutation = []
            for perm in distinct_permutations(positions):
                code = ""
                for idx, ch in enumerate(''.join(perm)):
                    # If the bit in the position is '1', then change the current bit in code to its opposite one
                    if ch == '1':
                        code += ('1' if query_code[idx] == '0' else '0')
                    else:
                        code += query_code[idx]
                permutation.append(code)
            all_permutations.append(permutation)
        return all_permutations


class HashTable:
    def __init__(self, hash_size, feature_size, W_parameter = 0.50):
        """
            Initializing necessary parameters for hash table.
            Hash method is based on:
                h(v) = (a * v + b) / W
            a is a vector that has the same dimension as the feature size, and
            b is the offset where b ⊂ [0, W]

            Arguments:
                - hash_size: number of hash function in the layer
                - feature_size: the dimension of the feature
                - W_parameter: the range of parameter b
        """
        self.hash_size = hash_size
        self.feature_size = feature_size
        self.W_parameter = W_parameter
        self.hash_table = collections.defaultdict(list)
        self.projections = self.get_random_projections(self.hash_size, self.feature_size)
        self.b_offsets = self.get_random_b_offsets(self.W_parameter, self.hash_size)

    @staticmethod
    def get_random_projections(hash_size, feature_size):
        """
            Get samples from normal distribution and make each row vector to unit length in order
            to speed up later processing
        """
        projections = np.random.randn(hash_size, feature_size)
        for index, row in enumerate(projections):
            projections[index] = projections[index] / np.linalg.norm(projections[index])
        return projections

    @staticmethod
    def get_random_b_offsets(W_parameter, hash_size):
        return np.array([np.random.uniform(0, W_parameter) for _ in range(hash_size)])


    def generate_hash(self, input_vector):
        """
            Perform dot product between each row in projection matrix and the input vector, and scale the position through b and W parameters. Next, check whether the value is above 0, if not then assign the value as 0 else assign 1.
        """
        code = ''
        for index, row_projection in enumerate(self.projections):
            hash = (((np.dot(input_vector, row_projection.T) + self.b_offsets[index])) / self.W_parameter)
            bool = (hash > 0).astype('int').astype('str')
            code += bool
        return code

    def get_bucket_objects(self, hash_value):
        return self.hash_table[hash_value]

    def __setitem__(self, input_vector, label):
        hash_value = self.generate_hash(input_vector)
        self.hash_table[hash_value].append(label)

    def __getitem__(self, input_vector):
        hash_value = self.generate_hash(input_vector)
        return self.hash_table[hash_value]
