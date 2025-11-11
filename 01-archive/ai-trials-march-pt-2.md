### Preface

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is 2 of 2 posts for the month of <u>March</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project.*

Let's recap. I originally planned to work with different tones this past month, but with the challenges I encountered in the prior month and the approach of Holi, a celebration I've been looking forward to, I invited Leonardo and StableDiffusion to join the party, with special appearances by Midjourney v5.2. I was pleasantly surprised with the images for Spring Festival, and equally horrified by the results that received for Holika Dahan. It felt as if all the advantages I had seen in the models in a previous iteration had disappeared, only to be replaced by additional issues.

After overcoming the disappointment of the results and concern that the images for Holi would be a disaster, I reflected upon prior experiences and concluded that they hadn't performed as poorly as I felt. Rather, I've become a bit spoiled with the alpha release of Midjourney v6.

- Leonardo did great with close-ups and small groups of people.
- Midjourney v5.2 did reasonably well in general, maintaining its position as the bar for its peers.
- StableDiffusion was fine as long as the subject matter wasn't human.

Additionally, I hadn't put in the work to take full advantage of Leonardo or StableDiffusion, especially when it comes to the need for negative prompts in the latter. All of this aside, I still question the ethics of pay-as-you-go platforms that produce questionable results as a default.

## Holi

### Article Creation

To maximize the number and variety of descriptions for Holi, I invited Gemini back to the party with Claude and ChatGPT. I had each of the three AI write its own article, followed by the descriptions for their respective articles.

Gemini's article exhibited the same shortcomings as prior articles on well-defined holidays. Rather than fighting to get the desired end result, I spoon-fed it step by step, having it write each article section one by one until I could assemble the entire article. Afterwards, I had it review the article for continuity, which is farther than I got with Bard. I had to nudge it along to complete the process, but it wasn't too challenging.

All three AI provided solid image descriptions with the current photographer role, although both Claude and Gemini needed a nudge regarding word count. I did some research to identify a more reliable way to get accurate word counts, and found I'm not alone in that quest. It's something to consider for future projects.

### Image Creation

With the image descriptions ready, I used Leonardo.ai's Kino model, Midjourney versions v5.2 and v6 alpha, and StableDiffusion's SDXL 1.0 model to generate images for each description in each article. As a result, I had hundreds of images to review and select the best ones for the articles. While it may be somewhat unfair to compare MJv6 against the others, I couldn't ignore its potential to produce the highest quality images for Holi.

For **Midjourney**, I provided the original descriptions without any edits or specialized parameters, only defining the aspect ratio.

*Images from Midjourney can be found within their respective articles, as well as **<u>the gallery for Holi</u>**.*

I had intended to research optimizing prompts for **Leonardo**, but the readily available guidance focused mainly on the built-in prompt creator. To supplement this and considering that most images featured people, I used PhotoReal v2, a recently released model.

To improve the results from **StableDiffusion**, I read about creating effective prompts for it. This led to notably better outcomes, but I clearly need to invest more effort to match the quality I've seen others achieve. I started with a smaller prompt and gradually expanded it to the following as I made updates along the way.

```
ugly | | bad face | | strange face | | disfigured | | misshapen face | | poorly drawn | | extra limbs | | extra hands | | extra feet | | backwards limbs | | extra fingers | | extra toes | | unrealistic, incorrect, bad anatomy | | cut off body pieces | | strange body positions | | impossible body positioning | | Mismatched eyes | | cross eyed | | crooked face | | crooked lips | | unclear | | undefined | | mutations | | deformities | | duplicate faces, plastic
```

A vast number of the issues were resolved with this basic negative prompt, but I have considerable work to do if I intend to uncover how far I can take SD.

## Your Own Worst Enemy

In my haste to generate Holi visuals, I neglected the most important lesson from my AI experiments this year: there is no magic bullet. AI, like any tool, requires effort to achieve optimal results. This holds true for content creators, designers, and virtually everyone else. I produced countless images that went straight to the digital trash bin. It was a sobering reminder that while AI is incredibly powerful, it's not a panacea. To get the desired output, you must carefully consider your input, refine through multiple iterations, and be prepared to learn from your missteps along the way.

Nevertheless, I witnessed significant advancements through the new Claude models, less remarkable improvements with Gemini, and of course, the substantial difference between Midjourney v6 and the other AI I used to create images this month. With more releases on the horizon, it promises to be an interesting year.

## Key Insights

**Positive Observations:**

- The process of guiding Gemini to produce an article highlighted the AI's ability to create coherent content with step-by-step instructions.
- Research into effective prompt creation for StableDiffusion led to improved outcomes, demonstrating the value of tailored prompts.

**Encountered Challenges:**

- The need for negative prompts in StableDiffusion and the lack of effort in optimizing prompts for Leonardo resulted in subpar image quality.
- The pay-as-you-go model of some AI platforms raises ethical concerns regarding the default quality of generated content.