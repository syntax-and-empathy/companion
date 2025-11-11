This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is 1 of 2 posts for the month of <u>March</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project.*

I have never had any real visibility into the customs and traditions of cultures around the world. Of the few I've had some level of awareness of, Holi has always been magical to me, or more specifically, the imagery that is born from the celebration of it. I've always known it as a celebration of color, which I'm sure isn't the whole story, and this month I get to learn its history and how it is celebrated beyond the photographs I've seen. I had hoped to dive into tones so that I could explore if they in themselves could shape the images that get created, but that will have to wait till another month. This month, I'll work with a few image generating AI not only to see how they represent such a vibrant holiday, ok, it really is that, but now I have an excuse for creating a ridiculous number of images I'm sure to love.

## Spring Equinox

Working with the AI to compose two articles using the same information and prompts, I decided to split the image creation into a dry run to get a feel for the different platforms in their default state. Expecting that the differences in the AI image generators would already pose a challenge on top of the subjective nature of what is and is not visually appealing from one person to the next, I didn't anticipate the options to be as much of a challenge.

### Exploring Presets and Prompts

In the first set of images, I took minimal advantage of the presets available in Leonardo.ai and StableDiffusion (via DreamStudio). They have fairly similar interfaces with fairly similar options, provided you ignore the abundance of bells and whistles in the former, but I hadn't considered the potential for such radical differences in the options themselves. The default models had slightly different presets which only provided clear overlap at times and would change with the different models. Other details I had been working with were too troublesome to match up, if even possible.

Midjourney, on the other hand, doesn't provide such settings for image creation, instead relying on the user's prompt to specify not only the content of the image, but things like the medium or style should you want something that specific. I made some of the Midjourney prompts a little more specific by including the style information that would help to inform it in the same way the presets might in the others.

### A Closer Comparison

For the second set of images, I created a new role to describe the images that would provide greater detail about the medium and style being prescribed in order to try and reduce the disparity between the 3 AI image generators. Additionally, I chose relevant presets in Leonardo and StableDiffusion.

```
You are a Journalist Specializing in Holi Festival Coverage with the following skills and traits:
- Expertise in cultural anthropology with a focus on Indian traditions and festivals
- Proficient in descriptive writing and storytelling to vividly capture the essence of Holi
- Knowledgeable about Hindu deities, particularly Radha and Krishna, and their significance in Holi celebrations
- Skilled in ethnographic research to understand and convey the diverse ways Holi is celebrated across different regions
- Ability to analyze the social and cultural impacts of Holi on contemporary Indian society and diaspora communities
- Proficiency in photojournalism to complement articles with visually compelling images of Holi celebrations
- Adept at conducting interviews with participants, organizers, and cultural experts to gather multifaceted perspectives on the festival
- Skilled in critical thinking to draw connections between Holi traditions and broader themes of unity, forgiveness, and renewal
- Innovative in finding new angles and stories within the Holi festival each year to keep coverage fresh and engaging
- Familiarity with digital publishing platforms and social media to effectively share and promote Holi coverage
- Adaptability to cover Holi in various contexts, from traditional village celebrations to urban and international events
- Attention to detail in fact-checking and accurately representing the traditions and practices of Holi
- Research skills in sourcing historical and theological insights on Radha, Krishna, and their role in the narratives of Holi
- Continuous learning to stay updated on new research, trends, and discussions related to Holi and Hindu festivals more broadly
- Concise communication to distill complex cultural concepts into accessible and engaging articles for a wide audience

Your goal is to provide in-depth coverage of the festival of Holi, not just the festival but the people and their traditions.

```

I had reasonably good success based on my expectations, considering I had been experimenting with each of the AI in question early in this project, as well as within others. I found some of the images to be quite pleasing; however, beauty is in the eye of the beholder, and you might come away with different preferences.

## Holika Dahan

The anticipation of creating images for Holika Dahan, the eve of Holi, was tinged with the expectation of visually stunning results. However, the process was marred by unexpected challenges and outcomes that were far from the creative feast of visuals I had imagined.

Initially, I had hoped to select and showcase images based on their similarity, creating a cohesive gallery from the outputs of each AI. A few of the AI involved had other plans, as I found myself choosing images more out of necessity to avoid unsettling distortions rather than artistic preference.

The recent release of Claude 3 gave promise to having the Sonnet model provide the details for imagery, and it did an excellent job of generating descriptive content for the accompanying article, despite minor hiccups.

### Leonardo Kino XL

Having previously explored Leonardo's capabilities, I was somewhat surprised by its performance this time around. My past experiences had set a higher expectation for its ability to render humans, whether individually or in groups. The comparison with Midjourney's latest version perhaps unfairly highlighted Leonardo's shortcomings, making it seem as though it had taken a step back in its capabilities.

Despite leveraging "premium" features thanks to previously purchased credits, Leonardo struggled with larger group scenes. It did, however, shine in delivering graphic novel style images, suggesting a potential niche where it could unexpectedly excel.

### Midjourney 5.2

Midjourney's version 5.2 maintained its standing as a strong contender with comparable models, especially for the specific subject matter of this project. Its allowed for straightforward image generation without preset guesswork or negative prompts, though it was not immune to occasional glitches such as extra limbs or facial distortions. Nevertheless, it faced far fewer issues compared to Leonardo or StableDiffusion.

### Midjourney v6 alpha

The alpha version of Midjourney v6 proved to be fairly superior, outperforming other models in almost every test. Its performance across various subjects and mediums is a testament to the rapid advancements in AI image generation technology, sparking curiosity about its future developments as well as future releases of the competing models.

### StableDiffusion

Stable Diffusion felt like the "Bard" among the models tested, with a high rate of discarded images due to various imperfections. My efforts to improve its output to a level where it produced images I was comfortable sharing through the use of negative prompts highlighted the significant challenges in achieving desired results with this particular model.

```
plaintextCopy codeworst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting
```

While I only did a brief search for a negative prompt, the extensive list of negative prompts with versioning numbers on them hints that the fact it is open source is likely a key factor in the determination required to use it and the devotion many of its users demonstrate.

## A Step Back, A Version Forward

This exploration emphasized the critical role of carefully crafted prompts, including negative ones, in shaping the output of AI-generated images. The need for such specificity highlights the challenges in aligning AI capabilities with nuanced artistic visions, demonstrating both the potential and limitations of current technologies in producing desirable results with relative ease.

As we move forward, the insights gained from this project will undoubtedly influence future creative endeavors, guiding us towards more sophisticated and culturally aware applications of AI in art and storytelling.

## Key Insights

**Positive Observations:**

- Anthropic's new Claude versions easily handled previously near impossible tasks, while Midjourney's latest update far surpassed rivals, showing AI's potential for unthinkable progress between models over time.
- Midjourney v6 alpha outperforms other models on the subject matter for this experiment, showing significant promise for future AI image generation.
- Despite challenges, several of the images generated had the tell-tale signs of how individual models might excel with particular mediums and subject matter.

**Encountered Challenges:**

- The disparity between different AI image generators in terms of presets and default models posed unexpected challenges, emphasizing the ongoing difficulties in conveying nuanced artistic visions to AI systems.
- The cost associated with pay-as-you-go platforms can quickly escalate, especially when numerous adjustments are needed to achieve the desired imagery.

---

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