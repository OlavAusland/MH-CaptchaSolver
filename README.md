# MH-CaptchaSolver
A custom captcha solver for the webgame called "[MafiaensHevn](https://mafiaenshevn.com)"

Using these two DNN there is no need for preprocessing, no color masking, convolution etc. This is achieved by creating a dataset
where I mask out everything except the characters (using hue color masks), turn it grayscale then apply an algorithm for finding the bounding boxes of each character
with some arbitrary threshold for the minimum area then sending this information to a folder and a bbox.json file.

### Automatically generating dataset
Furthermore, when this is deployed on the website, each captcha it solves is saved to a folder as an image, with the corresponding
bounding box for each characters per image to a .json file.

<img style="height:10vw;" src="Graphics/ExampleCaptchaPreprocessed.png"></img>

## Example Prediction

<img style="height:10vw;" src="Graphics/ExampleCaptcha.png"></img>
<img style="height:10vw;" src="Graphics/ExampleCaptchaSolved.png"></img>
## Pipeline
![Pipeline](Graphics/Pipeline.svg)

## Captcha Bounding Box Prediction Model
![BoundingBoxPredictionModel](Graphics/bbox_model.png)

# Character Prediction Model
![BoundingBoxPredictionModel](Graphics/character_model.png)

## Project Structure
#### CaptchaSolver.py
This si a script which takes in a path ('--path') to an image and outputs a prediction. It has a '--verbose' option which outputs more information about each prediction step.

#### Example.py
Example file on how to use the two prediction models together to sucessfully predict and output the final prediction

#### DNN.py
File for handling everything from loading the dataset, creating and training the individual model.

#### CreateCharacterDataset.py
File for creating the character dataset which populates the 'character' folder of individual characters.
