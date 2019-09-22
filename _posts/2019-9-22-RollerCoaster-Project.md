---
layout: post
title: Rollercoaster Project
---
#### Algorithm to Rank Roller Coasters Worldwide

Over the course of the last two weeks, our machine learning class has been organizing and developing an algorithim to rank roller coasters based on given conditions. In order to create this ranking system, our group scored every roller coaster based on its height, length, speed, type, and number of inversions. However, we ran into trouble because each value is scaled differently. As a result, we created a formula that transforms each value into a scale of 1 which would make the values relative and comparable to eachother. 

This formula was the following: obvserved - maximum value divided by the range. 

```javascript
/* Socoring Algorithm */
Score = Operating.assign(Score = lambda x:
                 1.5 * ((Operating['Height (feet)'] - Operating['Height (feet)'].min())/(Operating['Height (feet)'].max() - Operating['Height (feet)'].min()))
               + 0.7 * ((Operating['Length (feet)'] - Operating['Length (feet)'].min())/(Operating['Length (feet)'].max() - Operating['Length (feet)'].min()))
               + 2.0 * ((Operating['Speed (mph)'] - Operating['Speed (mph)'].min())/(Operating['Speed (mph)'].max() - Operating['Speed (mph)'].min()))
               + 1.0 * ((Operating['Type'] - Operating['Type'].min())/(Operating['Type'].max() - Operating['Type'].min()))
               + 0.7 * (((Operating['Number of Inversions'] - Operating['Number of Inversions'].min())/(Operating['Number of Inversions'].max() - Operating['Number of Inversions'].min())))
                        )
```

<img src="/images/Rollercoaster.jpeg" width="800"/>