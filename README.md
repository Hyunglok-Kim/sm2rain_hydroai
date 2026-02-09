# SM2RAIN-HydroAI

An automated HydroAI framework for rainfall estimation from satellite soil moisture using the SM2RAIN family of algorithms, from the original formulation to the most recent research developments.

---

## 1. Overview

SM2RAIN-HydroAI provides a unified, reproducible, and operational-ready environment for SM2RAIN-based rainfall estimation. The repository supports automatic data acquisition, preprocessing, and execution of multiple SM2RAIN algorithm versions, enabling consistent hydrologic analysis across time, regions, and sensors.

This framework is intended for research, development, and government-level operational collaboration.

---

## 2. Purpose

This repository was developed to:

(1) Maintain up-to-date implementations of SM2RAIN algorithms  
(2) Automatically download and manage required satellite and auxiliary data  
(3) Provide a standardized pipeline for rainfall inversion from soil moisture  
(4) Support reproducible research and operational testing  
(5) Enable collaboration with agencies such as KIHS  

---

## 3. Algorithms Included

The repository integrates multiple generations of SM2RAIN:

(1) Original SM2RAIN algorithm proposed by **Lucca Brocca**  
(2) And advanced operational variants:
(2-1) SM2RAIN-NSF (Net-Water-Flux version)
(2-2) SM2RAIN-GA (Green–Ampt Version)  
(3) Experimental AI-assisted SM2RAIN extensions (optional modules)

Each version is modular and can be selected depending on research or operational needs.

---

## 4. Main Features

(1) Automatic satellite soil moisture data download  
(2) Automated preprocessing and quality control  
(3) Multi-version SM2RAIN execution framework  
(4) Parameter configuration management  
(5) Reproducible workflow structure  
(6) Designed for scaling to regional or national domains  

---

## 5. Typical Workflow

Step 1 - Configure region, time range, and dataset  
Step 2 - Run data download module  
Step 3 - Execute preprocessing pipeline  
Step 4 - Select SM2RAIN algorithm version  
Step 5 - Run rainfall estimation  
Step 6 - Generate outputs for hydrologic applications  

---

## 6. Target Applications

(1) Rainfall estimation in data-scarce regions  
(2) Hydrologic model input generation  
(3) Flood and flash drought monitoring  
(4) Water cycle analysis  
(5) Operational testing for national water agencies  

---

## 7. Intended Users

This repository is structured for:

(1) Hydrology researchers  
(2) Remote sensing scientists  
(3) AI-hydrology developers  
(4) Government water agencies  
(5) Operational hydrologic monitoring teams  

---

## 8. Reproducibility and Governance Use

SM2RAIN-HydroAI emphasizes:

(1) Transparent algorithm versions  
(2) Traceable parameter settings  
(3) Automated and standardized processing  
(4) Compatibility with operational workflows  
(5) Documentation suitable for institutional use  

---

## 9. Acknowledgment

This framework builds upon foundational SM2RAIN research initiated by **Lucca Brocca** and subsequent developments in the hydrology and remote sensing community.
