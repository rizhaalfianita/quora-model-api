import numpy as np 

class TFIDF:
    def __init__(self, dataset):
        self.dataset = dataset
        self.word_list = self.create_word_list()
        self.word_count_list = self.create_word_count_list()
        self.tf_matrix = self.create_tf_matrix()
        self.idf_vector = self.create_idf_vector()
        self.transform = self.transform_tfidf()

    def create_word_list(self):
        word_list = []
        for sentence_array in self.dataset:
            for word in sentence_array:
                if word not in word_list:
                    word_list.append(word)
        return word_list

    def create_word_count_list(self):
        word_count_list = {}
        for term in self.word_list:
            word_count_list[term] = 0 
            for sentence_array in self.dataset:
                for word in sentence_array:
                    if term == word:
                        word_count_list[term] += 1
        return word_count_list
    
    def count_tf(self, sentence_array):
        tf_list = [0] * len(self.word_list)
        for word in sentence_array:
            for i in range(len(self.word_list)):
                if word == self.word_list[i]:
                    tf_list[i] +=1
        tf_list = np.array(tf_list) / len(sentence_array)
        return tf_list

    def create_tf_matrix(self):
        matrix_tf = []
        for sentence_array in self.dataset:
            matrix_tf.append(self.count_tf(sentence_array))
        return matrix_tf
            
    def create_idf_vector(self):
        idf_list = []
        for term in self.word_count_list:
            idf = np.log(len(self.dataset) / self.word_count_list[term])
            idf_list.append(idf)
        return idf_list

    def create_tfidf(self, sentence):
        tfidf_list = []
        for term in range(len(self.idf_vector)):
            tfidf = np.multiply(self.tf_matrix[sentence][term], self.idf_vector[term])
            tfidf_list.append(tfidf)
        return tfidf_list
    
    def transform_tfidf(self):
        tf_idf_vector = []
        for sentence in range(len(self.tf_matrix)):
           tf_idf_vector.append(self.create_tfidf(sentence))
        return np.array(tf_idf_vector)

if __name__ == "__main__":
    data = [["hello", "hello", "down", "there"], ["hello", "up", "there"], ["hello", "down", "there", "asd", "apa", "iya", "ahha"], ["hello", "up", "there"]]
    tfidf = TFIDF(data)
    tfidf_result = tfidf.transform
    print(tfidf_result.shape)
