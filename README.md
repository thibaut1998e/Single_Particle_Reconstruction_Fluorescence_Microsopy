# Single_Particle_Reconstruction_Fluorescence_Microsopy

# 🧬 README

## 📚 Context

"This repository contains the code developed during my PhD, which focuses on image processing techniques applied to structural biology—a field dedicated to understanding the three-dimensional (3D) structures of proteins and macromolecular assemblies. These assemblies are stable 3D complexes formed by multiple interacting proteins.

In recent years, **fluorescence microscopy** has emerged as a promising technique in structural biology. This imaging modality is a technique that uses light to excite fluorescent molecules in a sample, causing them to emit light of a different wavelength that can be detected to visualize structures. The field has seen major advancements thanks to the development of **super-resolution techniques** such as STED (Stimulated Emission Depletion), SIM (Structured Illumination Microscopy), and **expansion microscopy**, which physically enlarges samples to improve resolution.

## ❗ Problem Statement

Despite these advances, two major limitations hinder the use of fluorescence microscopy in high-resolution structural reconstruction:

1. **Anisotropic resolution**: The resolution is not uniform across all directions—typically lower along the optical axis (depth) of the microscope.
2. **Sparse labeling**: Fluorescent markers (fluorophores) used to tag proteins often label only part of the structure, leading to incomplete spatial information.

These limitations are intrinsic to the acquisition process and are difficult to overcome with physical improvements alone.

## 💡 Proposed Solution

This thesis aims to bypass these limitations using computational methods, specifically by developing a **particle-based reconstruction method** from isolated views, inspired by techniques used in cryo-electron microscopy (cryo-EM). The goal is to computationally combine multiple 3D views with **anisotropic resolution** of identical biological particles—captured in unknown orientations—to reconstruct a 3D model of the particle.

By fusing complementary information from each view, the method aims to generate a 3D volume with **isotropic resolution** and **uniform labeling density**, overcoming the physical constraints of the original acquisition process. This work focuses particularly on **convolutional imaging modalities** (which means that the acquired images are modelled as the convolution of the original object with the point-spread-function of the microscope)

### 🔄 Reconstruction Pipeline

The proposed reconstruction method consists of three main steps:

1. **🔍 Detection**: Starting from images containing many randomly oriented particles, the method detects the 3D coordinates of image patches (views) corresponding to individual particles.

2. **🧪 Ab-initio Reconstruction**: This step jointly estimates the 3D volume and the orientations (poses) of the views. It starts from a random initialization, requiring no prior knowledge about the structure or the poses. This forms a **blind inverse problem**, aiming to infer the hidden structure that produced the observed views.

3. **🎯 Refinement**: Using the ab-initio model as initialization, a refinement procedure is applied to obtain a more accurate estimate of both the 3D volume and the poses.

## 🧩 Homogeneous vs. Heterogeneous Reconstruction

Reconstruction from anisotropic views can be classified as:

- **Homogeneous Reconstruction**: Assumes all views originate from a single, fixed 3D structure. This was the focus of **Chapter 3** of the thesis.

- **Heterogeneous Reconstruction**: Recognizes that the biological particle may exist in multiple conformational states (structural heterogeneity). In this case, the goal is to reconstruct multiple volumes and to assign each view to the corresponding conformation. This extension is covered in **Chapter 4** of the thesis.
