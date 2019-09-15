
import tensorflow as tf

a = tf.constant(5) 
b = tf.constant(4)
c = tf.multiply(a,b)

d = tf.constant(2)
e = tf.constant(3)
f = tf.multiply(d,e)

g = tf.add(c,f)

with tf.Session() as sess:
  writer = tf.summary.FileWriter("output",sess.graph)
  print(sess.run(g))
  writer.close()


#tensorboard --logdir=output --port=6003

