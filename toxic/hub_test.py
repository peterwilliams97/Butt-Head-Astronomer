import tensorflow as tf
import tensorflow_hub as hub


embed = hub.Module('https://tfhub.dev/google/nnlm-en-dim50/1')
embeddings = embed(['The movie was great!'])

print('embed:', type(embed))
print('embeddings', type(embeddings))


with tf.Graph().as_default():
    # embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")
    # embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
    embed = hub.Module('https://tfhub.dev/google/nnlm-en-dim50/1')
    embeddings = embed(['The movie was great!'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        print(sess.run(embeddings))
