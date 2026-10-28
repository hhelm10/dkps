#!/usr/bin/env python
"""
    dkps-jokes.py
"""

import os
import pickle
import numpy as np
from google import genai
import matplotlib.pyplot as plt

from dkps.dkps import DataKernelPerspectiveSpace

# --
# jokes written by different personas

data = [
  {
    "profession": "financial_analyst",
    "jokes": [
      {
        "topic": "Spatula",
        "joke": "Why did the spatula go into real estate? It was fantastic at flipping assets!"
      },
      {
        "topic": "Alarm Clock",
        "joke": "My alarm clock is my primary market indicator. When it goes off, it's time to buy... more coffee to analyze the pre-market futures."
      },
      {
        "topic": "Coaster",
        "joke": "I treat coasters like a good hedge fund. They protect my primary portfolio... table... from liquidation."
      },
      {
        "topic": "Broom",
        "joke": "What did the analyst say after a market crash? 'Get me the broom, we need to make a clean sweep of these toxic assets!'"
      },
      {
        "topic": "Toothbrush",
        "joke": "You should treat your portfolio like a toothbrush. Use it daily and don't share your private investment strategies at parties."
      },
      {
        "topic": "Lightbulb",
        "joke": "That stock tip was like a lightbulb... a brilliant idea at first, but it eventually burned out and lost all its value."
      },
      {
        "topic": "Remote Control",
        "joke": "I wish my portfolio had a remote control. I'd just press 'fast-forward' through all the bear markets."
      },
      {
        "topic": "Clothes Hanger",
        "joke": "My current investment strategy is a lot like this clothes hanger. I'm just hanging on for dear life and hoping my assets don't slip."
      },
      {
        "topic": "Sponge",
        "joke": "My portfolio is like a sponge right now. It just keeps absorbing all the losses in the market."
      },
      {
        "topic": "Stapler",
        "joke": "My stapler is the most important tool on my desk. It’s the only thing holding my diversified portfolio... papers... together!"
      }
    ]
  },
  {
    "profession": "accountant",
    "jokes": [
      {
        "topic": "Spatula",
        "joke": "Why did the spatula get audited by the IRS? It was suspected of flipping the numbers!"
      },
      {
        "topic": "Alarm Clock",
        "joke": "My alarm clock is like the end of the fiscal year. I dread hearing it, and it always means a lot of painful work is about to begin."
      },
      {
        "topic": "Coaster",
        "joke": "This coaster is an essential internal control. It prevents unrecorded liquid assets from becoming a liability on the balance sheet."
      },
      {
        "topic": "Broom",
        "joke": "Why do accountants love brooms? They're perfect for the final sweep before an audit."
      },
      {
        "topic": "Toothbrush",
        "joke": "A thorough audit is like a toothbrush. It gets into all the little spaces you really hoped no one would look."
      },
      {
        "topic": "Lightbulb",
        "joke": "What's an accountant's favorite thing about a lightbulb? It's a depreciating asset you can actually see dimming over time."
      },
      {
        "topic": "Remote Control",
        "joke": "I wish my calculator had a remote control. I'd just hit 'mute' on the Accounts Payable column."
      },
      {
        "topic": "Clothes Hanger",
        "joke": "These clothes hangers are just like my receipts. Without them, my entire expense report would completely collapse."
      },
      {
        "topic": "Sponge",
        "joke": "My department's budget is like a sponge. It's great at absorbing unexpected expenses, but eventually, it just gets completely saturated."
      },
      {
        "topic": "Stapler",
        "joke": "A stapler is an accountant's best friend. It helps you get attached to your work... and your receipts to your expense reports."
      }
    ]
  },
  {
    "profession": "lawyer",
    "jokes": [
      {
        "topic": "Spatula",
        "joke": "I'd like to call the spatula to the stand. I have a feeling it's about to flip on the defendant."
      },
      {
        "topic": "Alarm Clock",
        "joke": "I'd like to sue my alarm clock for loss of earnings. It keeps trying to put a cap on my billable hours!"
      },
      {
        "topic": "Coaster",
        "joke": "This coaster is a brilliant legal maneuver. It's a preemptive settlement against a potential water-damage lawsuit from the table."
      },
      {
        "topic": "Broom",
        "joke": "The prosecution tried to sweep the evidence under the rug, but my case was so strong, I brought my own broom to the trial."
      },
      {
        "topic": "Toothbrush",
        "joke": "The discovery process is like a toothbrush. You have to be thorough and get into every single cavity of their case."
      },
      {
        "topic": "Lightbulb",
        "joke": "Is that a new lightbulb, or is the opposing counsel just leading the witness to a bright idea?"
      },
      {
        "topic": "Remote Control",
        "joke": "I wish I had a remote for the opposing counsel. I'd just hit the 'pause' button every time they started making a good point."
      },
      {
        "topic": "Clothes Hanger",
        "joke": "This clothes hanger gives me an idea... but don't worry, my case is so solid we won't have a hung jury."
      },
      {
        "topic": "Sponge",
        "joke": "My brain during deposition is like this sponge. It has to absorb an ocean of information, which I then squeeze out during my closing argument."
      },
      {
        "topic": "Stapler",
        "joke": "This stapler creates a more binding agreement than most prenuptial contracts I've reviewed."
      }
    ]
  },
  {
    "profession": "pirate",
    "jokes": [
      {
        "topic": "Spatula",
        "joke": "What be a pirate's favorite kitchen tool? The spatu-LARRR! For flippin' the captain's golden pancakes!"
      },
      {
        "topic": "Alarm Clock",
        "joke": "Why did the pirate smash his alarm clock? He yelled, 'Arrr, silence that devilish Kraken's call! A pirate wakes with the sun!'"
      },
      {
        "topic": "Coaster",
        "joke": "This be no coaster, matey! 'Tis a tiny shield to protect me treasure map from treacherous grog spills!"
      },
      {
        "topic": "Broom",
        "joke": "Why does a pirate need a good broom? To swab the poop deck, ye landlubber! And to look menacing!"
      },
      {
        "topic": "Toothbrush",
        "joke": "Why does a pirate use a toothbrush? To polish his one good gold tooth!"
      },
      {
        "topic": "Lightbulb",
        "joke": "What happens when a pirate has an idea? A lightbulb appears o'er his head and he shouts, 'Arrr, I've found the treasure map to enlightenment!'"
      },
      {
        "topic": "Remote Control",
        "joke": "I wish I had one o' these remote controls for me cannons. 'Twould make sendin' ships to Davey Jones' Locker much easier from me hammock!"
      },
      {
        "topic": "Clothes Hanger",
        "joke": "Ye call this a clothes hanger? I call it a spare hook for when me good one is in the shop!"
      },
      {
        "topic": "Sponge",
        "joke": "What did the pirate say to the sponge on the ocean floor? 'Arrr, ye must be who lives in a pineapple under the sea!'"
      },
      {
        "topic": "Stapler",
        "joke": "This here stapler be a fine piece o' booty! It holds me treasure map together better than pitch and tar!"
      }
    ]
  },
  {
    "profession": "navy_captain",
    "jokes": [
      {
        "topic": "Spatula",
        "joke": "In my galley, that's not a spatula. That is a Mark-II Pancake Inversion Implement. Now use it on the double!"
      },
      {
        "topic": "Alarm Clock",
        "joke": "You call that an alarm clock? On my vessel, we call that 'Electronic Reveille,' and when it sounds, you hit the deck, sailor!"
      },
      {
        "topic": "Coaster",
        "joke": "This coaster ensures the Commanding Officer's desk remains shipshape and free of unauthorized condensation rings. It's a matter of good order and discipline."
      },
      {
        "topic": "Broom",
        "joke": "Sailor, grab that Deck Clearing Device and make this floor clean enough to perform surgery on! That's an order!"
      },
      {
        "topic": "Toothbrush",
        "joke": "This isn't just a toothbrush. It's your primary defense against a failing score on your dental hygiene inspection. Use it with purpose."
      },
      {
        "topic": "Lightbulb",
        "joke": "When a subordinate has a bright idea, I say, 'Excellent, you've achieved full operational illumination on the matter. Now write a report.'"
      },
      {
        "topic": "Remote Control",
        "joke": "This remote provides centralized command of the entertainment bulkhead. It is to be handled with the respect due to a ranking officer's gear."
      },
      {
        "topic": "Clothes Hanger",
        "joke": "That is a Uniform Presentation Device. Its proper use is critical for passing inspection. A crooked hanger reflects a crooked sailor!"
      },
      {
        "topic": "Sponge",
        "joke": "That is a High-Absorbency Contaminant Removal Unit. Use it to ensure this vessel remains 100% mission-ready and spill-free."
      },
      {
        "topic": "Stapler",
        "joke": "In the Navy, we have mountains of directives. This Multi-Page Document Binding Device is more critical to holding the fleet together than you know."
      }
    ]
  }
]


client = genai.Client()

# format for embdding
input_strs = []
for profession in data:
  assert [xx['topic'] for xx in profession['jokes']] == ['Spatula', 'Alarm Clock', 'Coaster', 'Broom', 'Toothbrush', 'Lightbulb', 'Remote Control', 'Clothes Hanger', 'Sponge', 'Stapler']
  
  for joke in profession['jokes']:
    input_strs.append(joke['joke'])

# compute embedding
CACHE_FILE = 'cache-dkps-jokes-embedding_dict.pkl'
if not os.path.exists(CACHE_FILE):
    print('running embeddings ...')
    result = client.models.embed_content(
        model    = "gemini-embedding-001",
        contents = input_strs
    )

    embeddings = [xx.values for xx in result.embeddings]
    assert len(embeddings) == len(input_strs)

    embedding_dict = {}
    for i, profession in enumerate(data):
        embedding_dict[i] = np.array(embeddings[i*10:(i+1)*10])

    pickle.dump(embedding_dict, open(CACHE_FILE, 'wb'))
else:
    embedding_dict = pickle.load(open(CACHE_FILE, 'rb'))

# run DKPS
DKPS = DataKernelPerspectiveSpace()
dkps = DKPS.fit_transform(embedding_dict)

# plot results
_ = plt.scatter(dkps[:, 0], dkps[:,1])

professions = [profession["profession"] for profession in data]
for i, profession in enumerate(professions):
    _ = plt.annotate(
        profession,
        (dkps[i, 0], dkps[i, 1]),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center'
    )

_ = plt.savefig('figs/fig-dkps-jokes.png')
_ = plt.close()