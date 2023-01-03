# Kohls MLE Interview

This repo contains instructions, code and data for a take-home assessment.
Upon receiving this assessment, you should clone this repo to a local
environment with a [Python installation](https://www.python.org/downloads/).

This take home assignment is designed to assess following:

1. Generalize code to promote code reuse.
2. Configure code to switch parameters without changing code.
3. Demonstrate knowledge and experience in moving DS models from notebooks to production.
4. Prepare the Artifact to deploy in the server/compute VM.
5. Be prepared to discuss implementation of the model in a Cloud environment (Azure, GCP or AWS).

## Getting Started

Your starting point will be the [Model_Training.ipynb](notebooks/MLE_Interview_DS.ipynb) notebook provided in
the `notebooks` directory of this repo.

We ask that you:

1.  Take the model training code within the notebook
    and create a python codebase that accomplishes the following:

    - Trains the model by running a script (a `.py` file, bash or PowerShell script
      as you prefer).
      - The script should, at minimum, accept arguments or
        a configuration file that defines the data location, important model training
        parameters, and the model save location.
    - Log model metrics to an external tracking file/server/DB.
      - You can use your own implementation or integrate with an existing
        library like MLFlow, etc.
    - Save the trained model so that it can be reused later.

2.  Write a `Dockerfile(s)` which builds a container capable of scoring
    an array of data sent to the container in the body of a POST request.

    - Your container `build` step should accept an argument denoting the location
      of a a saved model to be used for scoring.
    - We should be able to build your Docker image locally and make a POST
      to a specified route (ie: `http://localhost:8888/score`) with an
      array of data and receive the model predicitons back.

3.  Be prepared to discuss how you would deploy your model training and scoring
    to your favorite Cloud service ( GCP, Azure, etc).

    - You can assume the training data is in your favorite Cloud File Storage service
      (ie: Azure Blob Storage, Google Cloud Storage, Amazon S3, etc).
    - You may write/leverage a CI/CD pipeline to demonstrate, or be prepared to
      talk about this step.
    - You may include a simple architecture diagram and talk through it or be prepared
      to whiteboard your ideas.
    - Be prepared to discuss integrations with relevant MLOps services on your chosen
      platform (ie: AzureML, VertexAI, Sagemaker, etc).

It is recommended that you push the resulting work to your own Git project and
be prepared to provide the link and discuss your work at the interview. The purpose of this
section is to provide a platform you the candidate to demonstrate familiarity with the technologies
and processes involved in migrating DS modeling from notebooks to production. We do not expect
perfect/professional code in the limited timeframe, but be prepared to use your submission to
demonstrate your capabilities in this area.

## Interview

During the follow-up interview we will ask you to present your
<b>resulting repo</b> that accomplishes the above. We expect to see
the above training and scoring scripts together with any helper
modules/classes you may have defined/written organized in a logical
manner consistent with your `Docker` implementations.

Some key questions we will work through:

- How might you improve the model? Different algorithm?
- How did you structure the training codebase? Why?
- How did you handle model metric tracking?
- How did/would you handle model versioning and updating?
- How did you handle environment replication/stability?
- How did/would you handle deployment to multiple environments (`dev`/`test`/`prod`)?
- How would you deploy this model to your favorite Cloud service? What services/architecture pattern would you use?
- What would change as the volume of training or scoring data increased dramatically?
- How would you begin writing a CI/CD pipeline in your favorite platform (Azure Dev Ops, Gitlab CI, GitHub Actions, etc.)
  to support your architecture and deployment?

**NOTE**: We do not expect all of the above questions to be worked-out/demonstrated
in your codebase, but you should be prepared to whiteboard/discuss these issues.
