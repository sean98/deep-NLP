class _DataLoader:
    def __init__(self, dir):
        self._dir = dir
        self._train = None
        self._dev = None
        self._test = None

    @staticmethod
    def _import_file(file_path):
        with open(file_path) as f:
            sentences = f.read().strip().split('\n\n')
            return [[tuple(line.strip().split('\t')) for line in s.strip().split('\n')] for s in sentences]

    @property
    def train(self):
        if self._train is None:
            self._train = _DataLoader._import_file(f'{self._dir}/train')

        return self._train

    @property
    def dev(self):
        if self._dev is None:
            self._dev = _DataLoader._import_file(f'{self._dir}/dev')

        return self._dev

    @property
    def test(self):
        if self._test is None:
            self._test = _DataLoader._import_file(f'{self._dir}/test')

        return self._test


NER = _DataLoader('ner')
POS = _DataLoader('pos')
