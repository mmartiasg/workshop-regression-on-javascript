# workshop-regression-on-javascript
This project was done to teach how regressions work to a group of students that were learning javascript. This was prior to the workshop before teaching neural networks.

The problem I had as an example to use is to predictor how good a wine is and there is a docker version with an api in docker hub that you can try here 
``` mmatiasg/node-dev:single-api ```

To use that image just do:
 - docker run -rm mmatiasg/node-dev:single-api
  ```That command will start up the api in port 3000```
 - then you can do:
  ```curl --request GET http://localhost:3000/predict\?alcohol\=8```
 
 To ask for a prediction

Observations:
 - It's a quite small dataset
 - The score goes from 3 to 8 instead of 0 to 10. There are not enough representations.
 - The baseline gets a validation score of 0.68
 - Using a simple regression with just 1 feature (alcohol) the validation score is 0.54.
  It's better than the baseline. But it will depend on how much that will impact the business. 
  If that is good enough or do we need to improve our model or maybe get better data?
  
  Questions to do once we get here are:
   - What is worst to give a lower score to a good wine or giving a higher score to a bad one
   - The error will impact the business in at least 2 ways:
    - If we give a bad wine consistently a good score. The business might loss some of its clients.
    - If we give a good wine a bad score and we use that score to price our wines we might end up losing money and bankrupt in the end...
    
   There is a trade-off there, the model performance is not enough to understand how well that solution will perform in a productive environment.
