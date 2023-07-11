# Fall Guys AI
This is a repo with my attempt to create a Fall Guys AI, that uses Behaviour Cloning in order to learn the game.

## Dataset
The project uses a custom dataset that was recorded by Adrian Necula, a very good player of Fall Guys.
In order to map the keys pressed with the actual frames, the recording was created using ```record_data.py```

The data for each recording is saved as a tuple of two: A video and a numpy array with the keys pressed.

## Model
The models tested were a **ResNet50** and a **VisionTransformer**.

## Data Handling
Because the images were saved as a video, we extract random sequences of frames directly from the video, and use
the asscoiated keys for their predictions.

The main class for handling images is **ImageDataLoader** from ```pipeline/image_data_loader```,
And the main class for handling videos is ** VideoDataLoader** from ```pipeline/video_data_loader```

## Training
The purpose of the models are to predict the next best move considering the last frame / last frames.

Due to the fact that there can be multiple keys pressed at once, each combination of keys is modeled as a state
and the model predicts directly the combination of keys.
