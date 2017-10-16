import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import train_pg as train_pg
import numpy as np

class MyTestCase(tf.test.TestCase):
  """ Test fixture for train_pg networks."""

  def testBuildMlp(self):
    """ Test that network builds."""
    n_features = 1
    x = tf.placeholder(dtype=tf.float32, shape = (n_features, 3))
    inner_net_size = 4
    output_net_size = 7
    net = train_pg.build_mlp(x, output_net_size, size = inner_net_size, scope="test_scope")
    self.assertListEqual([n_features, output_net_size], net.get_shape().as_list())

  def testRunBuildMlp(self):
    """Test training on the mlp net."""
    with self.test_session() as sess:
      with tf.variable_scope('scope', initializer=tf.ones_initializer()):
        shape = (1, 3)
        x = tf.placeholder(dtype='float32', shape = shape)
        output_size = 10
        net = train_pg.build_mlp(x, output_size, scope = 'scope')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        out = sess.run(net, feed_dict = {x : np.zeros(shape)})


if __name__ == '__main__':
  tf.test.main()
