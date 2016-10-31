'''
Created on 28 Jan 2016

@author: peng
'''
import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
print matrix1

matrix2 = tf.constant([[2.],[2.]])
print matrix2
product = tf.matmul(matrix1, matrix2)

#----------------------------------------------------------- sess = tf.Session()
#------------------------------------------------------------------------------ 
#---------------------------------------------------- result = sess.run(product)
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------ print result
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------ sess.close()

state = tf.Variable(0, name="counter")



input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})