import sys
from mrjob.job import MRJob, MRStep
import heapq


class TopNWords(MRJob):

    def mapper(self, key, line):
        if line.strip() != "":
            for word in line.strip().split():
                yield word, 1

    def combiner(self, word, counts):
        yield word, sum(counts)

    def reducer_sum(self, word, counts):
        yield None, (sum(counts), word)

    def top_n_reducer(self, _, word_cnts):
        for cnt, word in heapq.nlargest(2, word_cnts):
            yield word, cnt

    def steps(self):
        return [MRStep(mapper=self.mapper
                       , combiner=self.combiner
                       , reducer=self.reducer_sum)
            , MRStep(reducer=self.top_n_reducer)
                ]


def main():
    TopNWords.run()


if __name__ == "__main__":
    main()
