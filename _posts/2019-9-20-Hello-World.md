---
layout: post
title: Rollercoaster Project
---
#### Algorithm to Rank Rollercoasters Worldwide

Over the course of the last two weeks, our machine learning class has been organizing and developing an algorithim to rank rollercoasters based on given conditions. 

```javascript
/* Some pointless Javascript */
Score = Operating.assign(Score = lambda x:
                 1.5 * ((Operating['Height (feet)'] - Operating['Height (feet)'].min())/(Operating['Height (feet)'].max() -             Operating['Height (feet)'].min()))
               + 0.7 * ((Operating['Length (feet)'] - Operating['Length (feet)'].min())/(Operating['Length (feet)'].max() -                   Operating['Length (feet)'].min()))
               + 2.0 * ((Operating['Speed (mph)'] - Operating['Speed (mph)'].min())/(Operating['Speed (mph)'].max() -             Operating['Speed (mph)'].min()))
               + 1.0 * ((Operating['Type'] - Operating['Type'].min())/(Operating['Type'].max() -                                              Operating['Type'].min()))
               + 0.7 * (((Operating['Number of Inversions'] - Operating['Number of Inversions'].min())/(Operating['Number of Inversions'].max() - Operating['Number of Inversions'].min())))
                        )
```