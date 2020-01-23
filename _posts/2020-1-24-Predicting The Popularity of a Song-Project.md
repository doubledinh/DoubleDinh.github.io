---
layout: post
title: Audio Prediction Project
---
#### Algorithm to Predict The Popularity of a Song Based on Latent Features

Ethan and I are people who listen to music often, and we often get into debates with our friends about whether or not a song is good. So when we were searching for project ideas, I thought that there must be something about songs that makes us like them, something quantitative that can be analyzed. So we decided to create an algorithm that could read in a song, and based on several aspects of the song could then predict how good the song is. We decided that the best way to do this would be to create a data frame of song data and then run latent features with some rating system as the target in order to determine what features of the song are most important to how good it is. After some searching, we found that there were not easily accessible datasets that contained song files or song data, so we realized that we would need to read in songs on our own and analyze them in order to construct our data frame.

At first, we thought that we might need to construct our own algorithm to read in files, but then we discovered Librosa, which could read in .wav files and allow other programs to analyze the file. In order to get a .wav file of a song, we needed another plugin, called AudioSegment, which could read in an mp3 file and convert it into a .wav file. To test this, we downloaded an mp3 of Uproar, by Lil Wayne, and used AudioSegment to convert it to a .wav file (Uproar became our test song). Then, we used another plugin called vamp to read in the .wav file using a key called nnls-chroma, which was supposed to be able to create a chart that showed a depiction of the music, but wasn’t particularly useful for the project. Nonetheless, it was a good starting point. However, the key failed to work as Vamp claimed that the key did not exist. After some digging, we discovered that the key had somehow not been downloaded and that we would need to download it separately. The key did not download like the other plugins that we had installed (we used pip to install the others through the terminal, directly into anaconda). Instead, it came as a folder in the downloads, that we then had to determine where to relocate. On the [Vamp Plugins website](https://www.vamp-plugins.org/), the instructions said that we should put the folder in the vamp folder in our audio settings. However, the vamp that we had installed was not in the same directory as our local computer. Our teacher informed us that anaconda had a separate root file on our computers where the plugins were installed, so we found the root file and navigated to the plugins, where we began searching for Vamp in order to place the plugin in its correct place. However, after we had found the vamp folders and placed the plugin into them, it still failed. So we attempted to place it in the folder recommended on the website, but it still did not work. 

After spending a few days of experimentation and research, we discovered that vamp was not a typical python plugin, but rather an open-source plugin that accessed other plugins that people built for it and was run through terminal, not anaconda. So, we created a route within Jupyter Notebook that would access the plugins in a folder that we had labeled vamp in the Audio Plugins folder, at which point we were able to use the key. We tediously tested and learned the inner workings of the Vamp plugin, only to realize that Vamp could only analyze songs individually and at an extremely slow pace. In essence, Vamp runs and collects audio features using the following line: vamp.collect(data, rate, "(Whatever you key is)"). But how do we determine the key? Well, since Vamp was originally a batch command prompt file, vampy allowed us to run the following command which listed the available plugins: print(vamp.list_plugins()).

Below you can find the basic framework of AudioSegment (used to convert .mp3 to .wav) and Vampy (used to analyze the songs):

```python
src = 'MP3_Files/Yummy.mp3'
dst = 'wav_Files/Yummy.wav'

Example_song = AudioSegment.from_mp3(src)
Example_song.export(dst, format="wav")
```

```python
# Using librosa to create two new items named data and rate
data, rate = librosa.load('wav_Files/Yummy.wav')
```

```python
chroma = vamp.collect(data, rate, "silvet:silvet")
stepsize, chromadata = chroma["matrix"]
plt.imshow(chromadata)
```


```python
Beat = vamp.collect(data, rate, "beatroot-vamp:beatroot")
Chord = vamp.collect(data, rate, "nnls-chroma:chordino")
```

We then discovered Sonic Annotator, which was a tool that we could run vamp plugins through in order to analyze songs. In essence, Sonic Annotator is a command line program for batch extraction of audio features from multiple audio files. The basic idea is to abstract out the process of doing the feature extraction from the extraction methods, by using Vamp plugins for feature extraction. In the end, the output format is either .csv or .txt. When discovering Sonic Annotator, we originally thought that it was another Vamp plugin. We were hugely mistaken. Since sonic annotator is a command line program, Vampy could not find the plugin when we moved it into the directory we stored other plugins. Instead, we discovered that we simply had to find Sonic Annotator program, route Jupyter notebook to the file and run it as a terminal command.

Once we finally got sonic annotator working, we decided to test it on Uproar and Portland by Drake. Using the tempo plugin from the BBC Vamp Plugins, sonic annotator created a txt file that had the song name and the tempo of the song on a line. We realized that the file itself was not going to be useful, but fortunately we had just started to learn about how to read in .txt files in comp sci. So we set about creating a function that could read in the file and analyze it to create a list that contained lists of each song and its tempo. (insert picture of function below). In order to do this, we read in each line of the file, then split it based on the commas, and appended the main list with a list containing the first item of the split line (the name) and the second (the tempo). Once done, the function could read in the file and be easily turned into a data frame containing the necessary information.

We then decided to use Sonic Annotator with the Silvet plugins, which would give the information of every note in the song. Again, we tested it on Uproar and Portland, and got a .txt file with every line containing a different note and various information about that note, including what note it was and its duration. So we decided to create another function that could read in the .txt file and analyze the notes of each song. Again, we used the list of lists format, where we had a list containing lists of each song, which contained lists of the data on each note split by a comma. We then ran through each song and pulled data from each song, which we then stored in a new list of lists, with each list containing a song and its most important features. The features for each song in the list are the song name, the number of notes, the frequency of notes, the average length of the note, the average frequency (hz), the standard deviation of notes (hz), the most common note (hz), and how often the most common note occurs. We used the function to read the notes file for Uproar and Portland and created a data frame with the results. Once we established that the functions worked, we then moved on to the next step: constructing our data frame.

Below is a section of our code, focusing on the Sonic Annotator commands:

```python
# ! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -l
```

```python
# ! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -d vamp:bbc-vamp-plugins:bbc-rhythm:tempo --recursive /Users/ethandinh/Desktop/Machine\ Learning/Final_Project\ /MP3_Files -w csv --csv-stdout > Tempo
# ! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -d vamp:silvet:silvet:notes --recursive /Users/ethandinh/Desktop/Machine\ Learning/Final_Project\ /MP3_Files -w csv --csv-stdout > Notes
```

Our first idea for grabbing song data was to contact Spotify’s API. Talk about Spotify API
Once we created the Ana data frame, we needed to prune it so that we could run a latent features algorithm on it. We stripped away all aspects of the data frame that were not directly concerned with the Spotify ratings, such as the song URI, and then dropped the popularity rating and stored it in another list. Then, we ran the latent features algorithm on the data frame using the popularity as the target and found that ____ were the highest correlated features to popularity. Then, we converted ana into a matrix and ran a least squares transformation on it and the popularity rating. Once we had the least-squares transformation, we multiplied a random song by the transformation to see if the predicted popularity was similar to the actual popularity score, and it was. Then, we tested all the songs to find the average difference between the predicted popularity and the actual popularity, which was 6.78, a pretty good score (the popularity score was on a scale of 0 to 100).

<img src="/images/Spotify.png" width="800"/>