# SpaDes

Spacecraft Design tool. Takes as an input a list of payloads and an orbit and produces a design and estimated cost. 
Also included is a configuration tool which takes as an input a list of components and produces a configuration.
WIP: Produces a list of components and automatically generates a configuration.

Runnable code script is SpacecraftDesignMain.py. This picks a random payload and orbit and generates a design.

Modules communicate via JSON files for flexibility. May switch to transferring objects at some point for speed.

Must use python 3.10 or 3.9 to install tat-c
