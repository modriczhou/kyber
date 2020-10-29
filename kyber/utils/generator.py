# -*- coding: utf-8 -*-
#

import numpy as np

class DataGenerator(object):
    def __init__(self, examples, fields, batch_size):
        self.examples = examples
        self.batch_size = batch_size
        self._steps = len(self.examples) // self.batch_size + int(len(self.examples) % self.batch_size != 0)
        self._batches = []
        self._fields = fields

    def __len__(self):
        return self._steps

    def __iter__(self):
        raise NotImplementedError

class Generator4Clf(DataGenerator):
    def __init__(self, examples, fields, batch_size):
        super(Generator4Clf, self).__init__(examples, fields, batch_size)

    def __iter__(self, random=False, tf_flag=True):
        idxs = list(range(len(self.examples)))
        cur_batch, cur_size = [], 0
        if random:
            np.random.shuffle(idxs)
        for i in idxs:
            cur_batch.append(self.examples[i])
            cur_size += 1
            if cur_size == self.batch_size or i == idxs[-1]:
                batch_data = Batch(cur_batch, self._fields)
                if not tf_flag:
                    yield batch_data
                else:
                    # 根据Field整理iter的x,y
                    y_data = []
                    x_data = []
                    # print(batch_data)
                    for name, field in self._fields.items():
                        # print(name, field)
                        if not field.is_target:
                            x_data.append(getattr(batch_data,name))
                        else:
                            y_data.append(getattr(batch_data,name))
                    # 判断多输入多输出
                    batch_x = x_data[0] if len(x_data)==1 else x_data
                    batch_y = y_data[0] if len(y_data)==1 else y_data
                    yield batch_x, batch_y

                cur_batch, cur_size = [], 0

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d

class Generator4Seq(DataGenerator):
    def __iter__(self, random=False):
        raise NotImplementedError


class Batch(object):
    def __init__(self, batch_data, fields):
        self.name = None
        if batch_data is not None:
            self.batch_size = len(batch_data)
            self.fields = fields
            self.input_fields = [k for k, v in fields.items() if v is not None and not v.is_target]
            self.target_fields = [k for k, v in fields.items() if v is not None and v.is_target]

            for name, field in fields.items():
                if field is not None:
                    field_batch = [getattr(x, name) for x in batch_data]
                    setattr(self, name, field.process_batch(field_batch))
