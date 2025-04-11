#!/bin/bash
mkdir -p ~/bin
wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
chmod +x chromedriver
mv chromedriver ~/bin/
rm chromedriver_linux64.zip
export PATH=$PATH:~/bin