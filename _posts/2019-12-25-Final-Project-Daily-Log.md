---
layout: post
Title: Final Project Daily Log
---

Below is a log of my daily progress, including:

1.) What I worked on today

2.) What I learned

3.) Any road blocks or questions that I need to get answered

4.) What my goals are for tomorrow

**January 7th**

Today, Hugh and I re-established our goals for our final project. We determined that we want to find/explore a python plugin that can measure the beat and tempo of a certain song. After we determined our goal, we researched and found an interesting plugin called Vamp and Sonic Annotator. Together they work in unison to read and process a song. We then pip installed vamp and librosa. We began experimenting, but the fire drill halted our progress. As for goals for tomorrow, Hugh and I hope to continue experimenting with vamp and exploring the data that it is able to gather. We also would like to find a way to mass import songs to produce a dataset ready for NLP analysis. 

**January 8th**

Today, Hugh and I ran into our first coding error. At first, we realized that librosa, a python plugin used to read in music files, defaulted to .wav files. We had originally used an online converter, however, since this course is practically computer science, we decided to research various python plugins to convert mp3 files. After a few attempts at brew installing the correct files, we successfully converted the audio file. Next, we tried to continue with the vamp protocol to covert the audio file into a matrix through one of vamp's plugins. However, we ran into another error: we had not installed the correct plugin. Tomorrow, our goal is to work through the issues with the solution you have provided and see if that solves the issue. If it does not, then we will continue to troubleshoot, but if it does, we will continue to see what we can do with the matrix produced from sonic annotator. 

**January 9th**
WE SOLVED OUR MISTAKE. First, I realized that because my mac was updated to Mojave, Anaconda was sourced in the wrong location and thus pip install was never able to add plugins to my python notebook. Instead, it was downloading it onto my local python environment. Through trial and error, I discovered that there was an alias command that sourced all of pip's downloads into the local bin file instead of Conda's. I also discovered that my conda environment is specific to my user instead of my whole mac, making it rather difficult to find where Conda was installed. After redoing all of the pip installs, I ran the file and it turned out to be successful. Vamp converted the .wav audio file into a matrix that can hopefully be read by other Vamp plugins. Tomorrow, Hugh and I will continue to explore Vamp's capabilities and see whether or not we can find the latent features of the matrix produced. 


**January 10th**
Last night, I worked on discovering how the vamp python plugin worked. I found that it was a rather simple command that took the installed vamp plugins and ran them within python, outputting the plugin's results. After discovering what each plugin did, in class, I found that sonic annotator was a bash command that could be run in python to produce a CSV of the selected vamp plugin. Unfortunately, my computer crashed and I could not troubleshoot why sonic annotator was unable to create the file. Hugh and I will try to solve the issue over the weekend and continue experimenting on Monday. 

**January 13th**
Today, after a long period of troubleshoot, Hugh and I discovered how to use Sonic Annotator. With the plugin, we were able to export our data into a text file. We also discovered that we can read in any file in a given path and read all the data. From there, Hugh used his Comp-Sci knowledge to create a program that is able to transfer the text file to a data frame that will get analyzed. Tomorrow, Hugh and I will try to finalize all the plugins and export everything into a data frame. We also need to figure out how to rip songs from Spotify's API. 

**January 15th**
Today, Hugh and I worked towards completing a function that would take the matrix outputted into a text file and convert it to a data frame that can be analyzed. However, in the process of working through the function, we discovered that all of the matrices produced by the plugin was under one file and thus we had to split the data frame into so that they would correlate with the correct song. We got it working partially, but still, it is not complete. Our goal for the next several days is to complete the function and put everything into a data frame that can be analyzed. And, we still need to get a bot that would store information from Spotify's API. 
