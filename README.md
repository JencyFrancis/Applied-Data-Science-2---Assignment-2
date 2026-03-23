Overview:
This assignment builds and evaluates two deep learning models for classifying films by genre using the Multimodal IMDB dataset. The two models work independently — a CNN classifies films based on their poster images, and an LSTM classifies films based on their text overviews. The entire pipeline is built in TensorFlow and Keras inside a Google Colab notebook, taking advantage of GPU acceleration throughout. 

Dataset:
The dataset is Multimodal_IMDB_dataset.zip, which contains film data from the Internet Movie Database. It includes film posters in JPEG format and text overviews for each film, along with genre labels used as the classification target.
Important note on the repository: Only the CSV dataset file has been uploaded to the repository. The 7,897 JPEG poster images could not be included due to file size restrictions. To run the full pipeline, the original zip file needs to be downloaded separately and extracted into the working directory before running the notebook.

Assignment Structure:
The work is broken into four successive sections, each building on the last.

The first section is data processing, and it covers both modalities separately. For the posters, an image processing function is written and then the TensorFlow tf.data API is used to build an efficient, optimised pipeline that feeds images into the CNN. For the overviews, the text vocabulary is built by calling encoder.adapt() and another tf.data pipeline is constructed to handle the text sequences efficiently.

The second section is model definition. For the posters, a convolutional neural network is constructed and compiled by reading a predefined model summary — the correct layer settings, output shapes, and parameter counts are all inferred directly from that summary rather than being given explicitly. For the overviews, an embedding layer is set up and a Keras sequential LSTM model is built using the layers specified in the notebook.

The third section is training. A set of callbacks is defined to log the training process, and both models are then trained for a specified number of epochs. GPU acceleration in Colab is what makes this feasible within a reasonable time.

The fourth section is evaluation. A selection of example films is chosen, their posters are plotted, their overviews are printed, and both models predict their genres. These predictions are then discussed critically in the written report — looking at where the models succeed, where they fail, and crucially why that might be happening based on the nature of the input data and the model architecture.

Conclusion:
This analysis shows that LSTM models significantly outperform CNN models in classifying movie genres based on text from movie overviews. While movie posters contain genre-related visual cues, they can be ambiguous, whereas overviews provide clear, relevant information. There are instances, particularly with visually distinct genres where visual cues may be more reliable. Therefore, a comprehensive classification system should integrate both text and visual approaches.
