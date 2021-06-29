from mrjob.job import MRJob


class MRWordFreauencyCount(MRJob):

    def mapper(self, _, line):
        """拆分阶段"""
        # print(line)
        yield "chars", len(line)
        yield "words", len(line.split())
        yield "lines", 1

    def reducer(self, key, values):
        """汇合阶段"""
        yield key, sum(values)


if __name__ == "__main__":
    MRWordFreauencyCount.run()
