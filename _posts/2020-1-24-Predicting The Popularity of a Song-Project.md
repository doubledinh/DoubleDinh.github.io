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
! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -l
```

```python
! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -d vamp:bbc-vamp-plugins:bbc-rhythm:tempo --recursive /Users/ethandinh/Desktop/Machine\ Learning/Final_Project\ /MP3_Files -w csv --csv-stdout > Tempo
! /Users/ethandinh/Library/Audio/Plug-Ins/Vamp/sonic-annotator -d vamp:silvet:silvet:notes --recursive /Users/ethandinh/Desktop/Machine\ Learning/Final_Project\ /MP3_Files -w csv --csv-stdout > Notes
```

Our first idea for grabbing song data was to contact Spotify’s API. Since Spotify already had a rating system that could be retrieved, we just had to learn how to communicate with the Spotify API. First, we had to create a developer Spotify account in order to find our client ID and client secret. These two will act as the authentication required to communicate. Next, we pip installed Spotipy, a package that allows python to run terminal Spotify command. In essence, Spotipy acted as our gateway into Spotify's data frame. Next, we ran a code that allowed us to retrieve a token using our client ID and secret. With that token, we are now able to run the Spotify commands, pulling tracks, URIs, Names, ratings, and all the various audio features. What was interesting, however, was the fact that the output from the command was essentially, a dictionary containing dictionaries of lists. As a result, we created a program that striped the output and append the necessary data into a list that was later transformed into a data frame (through pandas) called ana. 

Below, we included a snippet of the code used to communicate with Spotify:

```python
scope = 'user-library-read'
token = util.prompt_for_user_token('ethan.dinh',scope,client_id='bd4a55649bd242cb94ceadaf98c1ff21',client_secret='67d9d6c64d024428a93bbd8f669b503f',redirect_uri='http://localhost:8888/notebooks/Final_Project%20/Music%20reader%20experiments.ipynb')
sp = spotipy.Spotify(auth=token)

playlist_id = 'spotify:playlist:37i9dQZEVXbMDoHDwVN2tF'
results = sp.playlist(playlist_id)

names = []
for entry in results['tracks']['items']:
    names.append(entry['track']['name'])
    
tracks = []
for entry in results['tracks']['items']:
    tracks.append(entry['track']['uri'])
    
ratings = []
for entry in results['tracks']['items']:
    ratings.append(entry['track']['popularity'])
    
analysis = []
for i in range(len(tracks)):
    analysis.append(sp.audio_features(tracks[i]))
    
popularity = pd.DataFrame({'rating':ratings, 'Names':names})
popularity = popularity.set_index('Names')
popularity = popularity.rename(index={'Sunflower - Spider-Man: Into the Spider-Verse': 'Sunflower'})
```

Once we created the Ana data frame, we needed to prune it so that we could run a latent features algorithm on it. We stripped away all aspects of the data frame that were not directly concerned with the Spotify ratings, such as the song URI, and then dropped the popularity rating and stored it in another list. Then, we ran the latent features algorithm on the data frame using the popularity as the target and found that energy was the highest correlated features to popularity. Then, we converted ana into a matrix and ran a least squares transformation on it and the popularity rating. Once we had the least-squares transformation, we multiplied a random song by the transformation to see if the predicted popularity was similar to the actual popularity score, and it was. Then, we tested all the songs to find the average difference between the predicted popularity and the actual popularity, which was 6.78, a pretty good score (the popularity score was on a scale of 0 to 100).

Below are the results from Spotify's data:

    (62.78186683534804, 'energy')
    (8.763829026087407, 'valence')
    (5.754218079936674, 'speechiness')
    (-4.589870200537435e-07, 'duration_ms')
    (-0.05924298705592834, 'mode')
    (-0.17499091666564215, 'tempo')
    (-0.19424044245721284, 'acousticness')
    (-0.781781722126988, 'time_signature')
    (-0.8792381028852305, 'key')
    (-4.30644117093755, 'loudness')
    (-15.290106547492408, 'liveness')
    (-17.85018053625503, 'danceability')
    (-38.43631962912502, 'instrumentalness')


After analyzing the Spotify data frame, we decided to analyze our own rating system to see how it compared to Spotify’s. However, Spotify’s API did not allow us to download mp3 versions of each song, so we used a youtube to mp3 converter to pull all of the songs from the global top 50 list on youtube and download them in mp3 format. Then, we ran all of the songs through the Silvet plugin through sonic annotator, then read in the file through the get_notes function in order to construct a data frame of the important features that we had decided upon. Once the data frame was constructed, we ran latent features with Spotify’s popularity rating as the target, and found that average length of note and average frequency of note correlated most with a high popularity score, while the average hz and standard deviation of hz barely correlated at all. We then ran the average error score calculator on our model and came up with a score of 5.031, an even better score than Spotify’s.

Below are the latent features for our model:

    (62.671211105070256, 'Average Length of Note')
    (10.14723994925517, 'Frequent Notes')
    (1.3040152514846177, 'time_signature')
    (0.553463819117075, 'key')
    (0.05093160869888281, 'Average Htz')
    (0.005351438087299862, 'Most Common Note')
    (8.38831033840526e-05, 'duration_ms')
    (-0.013426779509758388, 'Number Notes')
    (-0.01386938315184956, 'Standard Deviation of Htz')
    (-0.02561054246682852, 'Most Common Note')

We set out with the goal to create an algorithm that could take in the name of a song and predict how good the song would become. We managed to create a model that could read several aspects of a song and predict how good it is, provided that we have data on it. We managed to streamline the model so that if the data for the song was already in our data frame, we could simply type in the song’s name and it would return both the predicted popularity score for the song and pull the popularity score from Spotify to match it to. In the future, we would like to streamline the model even more so that a person could import an mp3 file, and it would then run through the Silvet algorithm, get read in and added to the data frame, which would then be recalibrated using the new song in addition to all of the old data frame, and return the song’s predicted rating, and its Spotify rating if it has one. Even further down the line would be to create an algorithm that can scan the web and download mp3 files automatically, which could then be automatically added to the data frame, which would further increase the accuracy of our model.

<img src="/images/Spotify.png" width="800"/>

You can find our code located [here](https://github.com/doubledinh/Final_Project)