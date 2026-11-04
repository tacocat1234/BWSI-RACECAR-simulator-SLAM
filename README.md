# Project Explanation

This code is my implementation of the optional SLAM bonus challenge for the BWSI RACECAR. 
It is mostly built off of the code I developed during my time at BWSI as an optional extension beyond the core curriculum (It was included as a potential thing you could work on, but never explicitly taught anywhere or expected). I worked on it primarily on my own time at home after class or whenever we had our quest log code ready and were waiting for our turn to go downstairs and test on the official track. I also worked with my teammate Viir on some parts, such as ICP.

The original repository upon which this current repository was based on with the full commit history of SLAM as I developed it during the time of the course can be found [here](https://github.com/tacocat1234/BWSI-RACECAR-T9/tree/main/racecar-student/QUEST_LOG/SLAM/sixian_versions) and [here](https://github.com/vsood28/BWSI-RACECAR-T9/tree/1ee4400cf6ca3f922bce88c753f3b385ce0a90b2/racecar-student/QUEST_LOG/SLAM) 

The first link is the final product at the end of the program including the commits for unit test creation, and the second link is to the commit with the code prior to file reorganization.

The primary changes with regards to the final version at the end of the course was creating a simulated “fake ROS” file to interface between the simulator and the original code I wrote through a simulated ROS interface, since the code for the course previously interfaced with the real ROS nodes of the RACECAR.
Most of the other changes were bugfixes, such as sign errors or conversion errors, as well as developing a complete SLAM test with pathfinding, something I did not have enough time to finish before the course ended.

The relevant code files for this project are found in this [directory](https://github.com/tacocat1234/BWSI-RACECAR-simulator-SLAM/tree/main/racecar-student/labs/slam_sim)

The remainder of this README is the original README of the RACECAR NEO platform

# racecar-neo-installer

Instructions for RACECAR Neo installation found [here](https://docs.google.com/document/d/1WnaPee1AA4GwqgW9r1K-RrU2VonvI3ZcDkW5oSuaSaY/edit#heading=h.uxbegx6o23ro)

Template repository for native RACECAR installation (Entry Point)

For Windows 10/11, macOS 10.15+ (Catalina+, >2020), and Linux (tested distros include Ubuntu 20+ & Debian 10+)

## Prerequisites

Python 3.9.7 installation

## How to install Racecar Neo

1. `cd` into the folder that you want to install your project files in
2. run the command `git clone https://github.com/MITRacecarNeo/racecar-neo-installer.git`
3. run the command `bash racecar-neo-installer/racecar-student/scripts/setup.sh`
    - Follow the prompts to enter your operating system and curriculum type per your course
    - Wait ~15 minutes as the libraries and dependencies install.
    - Close and re-open your terminal window after the installation is complete
4. run the command `pip3 install -r racecar-neo-installer/racecar-student/scripts/requirements.txt` to install dependencies.
5. run `racecar test` and `racecar sim demo.py` to verify installation is successful

## What does the racecar-neo-installer do?

racecar-neo-installer is a flexible and efficient way of installing the RACECAR Neo software onto your computer. The repository only contains a `scripts` folder inside a template `racecar-student` folder. Upon cloning the folder into your computer and running the `setup.sh` script, the compiler will ask you to input your computer's **operating system and curriculum type**.

Then, it will run through a series of **python library dependency installations**, **bash alias scripts**, and **git clone commands** to set up the Racecar Neo installation to your unique specifications.

Since the installation process slightly varies between operating systems and types of RACECAR curriculum, the installer strives to simplify the setup process (making it less painful to go through), and make the course more accessible to those with less technical experience.

## Legacy Information

RACECAR Neo is based on RACECAR-MN (Model Neo), the course version supported between the years of 2019-2022. RACECAR-MN development is no longer ongoing and is considered to be deprecated. More on RACECAR-MN and the outdated installation process can be found here: [RACECAR-MN Installation](https://mitll-racecar-mn.readthedocs.io/en/latest/gettingStarted/computerSetup.html#)
