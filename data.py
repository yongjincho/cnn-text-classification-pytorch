class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    RESERVED_TOKENS = [PAD_TOKEN, UNK_TOKEN]
    PAD_TOKEN_ID = RESERVED_TOKENS.index(PAD_TOKEN)
    UNK_TOKEN_ID = RESERVED_TOKENS.index(UNK_TOKEN)

    def __init__(self, filename):
        self._tokens = Vocabulary.RESERVED_TOKENS +\
                       [line.strip().split("\t")[0] for line in open(filename)]
        self._token_to_id_map = {t: i for i, t in enumerate(self._tokens)}

    def token_to_id(self, token):
        return self._token_to_id_map.get(token, Vocabulary.UNK_TOKEN_ID)

    def id_to_token(self, id_):
        return self._tokens[id_]

    def __len__(self):
        return len(self._tokens)


class ClassificationDataset:
    def __init__(self, filepath, vocab, batch_size):
        self._file = open(filepath)
        self._vocab = vocab
        self._batch_size = batch_size
        self._reset()

    def _fill_buffer(self, size):
        if not self._buffer:
            for line in self._file:
                label, sentence = line.split("\t")
                label = int(label.strip())
                sequence = [self._vocab.token_to_id(t) for t in sentence.strip().split()]
                self._buffer.append((label, sequence))
                if len(self._buffer) >= size:
                    break

            self._buffer.sort(key=lambda x: len(x[1]))
            self._buffer_iter = iter(self._buffer)

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        self._fill_buffer(self._batch_size * 1000)

        label_batch = []
        sequence_batch = []
        for label, sequence in self._buffer_iter:
            label_batch.append(label)
            sequence_batch.append(sequence)
            if len(label_batch) == self._batch_size:
                break

        if not label_batch:
            raise StopIteration

        max_length = len(sequence_batch[-1])  # sequence_batch is sorted.
        for i in range(len(sequence_batch)):
            sequence_batch[i] = sequence_batch[i] + \
                    [Vocabulary.PAD_TOKEN_ID] * (max_length - len(sequence_batch[i]))

        return {"sequences": sequence_batch, "labels": label_batch, }

    def _reset(self):
        self._file.seek(0)
        self._buffer = []
        self._buffer_iter = None
