import tensorflow as tf
from tensorflow.keras import Model as M
from tensorflow.keras import Input as I
from tensorflow.keras import Sequential
from hyr.network_base import BaseNetwork
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization

class Encoder(BaseNetwork):
    '''CNN神经网络'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_net(self):
        self.conv41 = Conv2D(filters=128, kernel_size=(4, 1), kernel_initializer='he_uniform')
        self.flatten41 = Flatten()
        self.conv14 = Conv2D(filters=128, kernel_size=(1, 4), kernel_initializer='he_uniform')
        self.flatten14 = Flatten()
        self.conv22 = Conv2D(filters=128, kernel_size=(2, 2), kernel_initializer='he_uniform')
        self.flatten22 = Flatten()
        self.conv33 = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_uniform')
        self.flatten33 = Flatten()
        self.conv44 = Conv2D(filters=128, kernel_size=(4, 4), kernel_initializer='he_uniform')
        self.flatten44 = Flatten()

        self.l1 = Dense(512, kernel_initializer='he_uniform')
        self.b1 = BatchNormalization()
        self.l2 = Dense(128, kernel_initializer='he_uniform')
        self.b2 = BatchNormalization()
        self.l3 = Dense(self.a_dim, 'softmax')
        self(I(shape=self.s_dim))
        self._initialize()

    def call(self, s):
        x = tf.concat([
            self.flatten41(self.conv41(s)),
            self.flatten14(self.conv14(s)),
            self.flatten22(self.conv22(s)),
            self.flatten33(self.conv33(s)),
            self.flatten44(self.conv44(s))
        ], axis=-1)
        x = tf.nn.relu(x)
        x = self.l1(x)
        x = self.b1(x)
        x = tf.nn.relu(x)
        x = self.l2(x)
        x = self.b2(x)
        x = tf.nn.relu(x)
        x = self.l3(x)
        return x

    @tf.function
    def train(self, s, a):
        '''
        主训练函数，采用交叉熵多分类目标损失
        params:
            s: 棋盘处理后的状态， 维度为[B, H, W, C]
            a: 动作标签， 维度为[B, ]
        return:
            loss: 这次训练的损失
        '''
        with tf.device(self._device):
            with tf.GradientTape() as tape:
                a_predict = self(s) # [B, A]
                loss = tf.keras.losses.sparse_categorical_crossentropy(a, a_predict) # (B, )
                loss = tf.reduce_mean(loss) # (1, )
            grads = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.global_step.assign_add(1)
            return loss

if __name__ == "__main__":
    net = Encoder(s_dim=[84,84,3], a_dim=4)
    net.save_model()  
    net.load_model()
