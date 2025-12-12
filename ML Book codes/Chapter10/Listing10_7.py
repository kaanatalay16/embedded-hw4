import tensorflow as tf

reduction_enum = tf.keras.losses.Reduction

y_true = tf.convert_to_tensor([[0., 1.], [0., 0.]])
y_pred = tf.convert_to_tensor([[2., 1.], [3., 0.]])

mse = tf.keras.losses.MeanSquaredError()
mse_reduction = tf.keras.losses.MeanSquaredError(reduction_enum.NONE)

result = mse(y_true, y_pred)
result_reduction = mse_reduction(y_true, y_pred)

print(result.numpy())
print(result_reduction.numpy())