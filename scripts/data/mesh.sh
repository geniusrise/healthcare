#!/usr/bin/env bash

curl -o desc.zip https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2024.zip
curl -o qual2024.xml https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/qual2024.xml
curl -o supp.zip https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/supp2024.zip

unzip desc.zip
unzip supp.zip
