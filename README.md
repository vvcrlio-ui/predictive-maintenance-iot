README
======

IoT Predictive Maintenance System
Industrial Milling Machine Failure Prediction

This repository contains machine learning models and tools for predictive
maintenance based on real sensor data from milling machines. The system
predicts equipment failures and provides interpretability analysis for
root-cause diagnosis. An optional Streamlit interface demonstrates
real-time monitoring and model explanation.

Documentation
-------------

Dataset:
  UCI AI4I 2020 Predictive Maintenance Dataset
  https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset

Features:
  - Binary failure prediction (Normal / Failure)
  - SHAP-based feature attribution for diagnostics
  - Interactive monitoring demo via Streamlit

Project structure overview:
predictive-maintenance-iot/
├── data/
├── src/
├── models/
├── app/
└── README.md

Installation
------------

Clone the repository:
  $ git clone https://github.com/<username>/predictive-maintenance-iot.git
  $ cd predictive-maintenance-iot

Create a virtual environment and install dependencies:
  $ python3 -m venv venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt

Running the demo:
  $ streamlit run app/dashboard.py

Usage
-----

Example (5-line code usage for model loading):

  from model import load_model
  model = load_model("models/xgboost.pkl")
  prediction = model.predict(sample)
  print(prediction)

License
-------

This project is released under the MIT License.
See the LICENSE file for details.

Authors
-------

Wan Xiang <email@domain>
Contributions and issues welcome via GitHub.

Support and Contact
-------------------

Report issues:
  https://github.com/<username>/predictive-maintenance-iot/issues

Project website (if available):
  https://github.com/<username>

Change Log
----------

v1.0.0  Initial public release: model pipeline, SHAP analysis, Streamlit UI.