# Qualitative translation and responsible AI review

This is an AI-assisted close reading of eight deliberately selected test examples shared by all four models. It is not an independent professional translator assessment, a random audit, or a statistical fairness study. Examples were selected for identity, religion, cultural names, numerical claims, uncertainty and sensitive content. Claims below concern the displayed outputs only; no model is declared unbiased or safe from these examples.

Verbatim output excerpts are limited to 240 characters for readability. The complete outputs remain in the prediction JSON files; an ellipsis added here marks display shortening, not a change to the evaluated text.

## Gender identity and negation

Test ID: `811de7767c5defc6` (zero-based row 5).

**English source:** One intersex person, according to the BBC, said years later, "I am neither a man nor a woman."

**Dataset reference:** बी.बी.सी. के अनुसार एक इंटरसेक्स व्यक्ति ने वर्षों बाद कहा था, "न तो मैं पुरुष हूँ, न महिला।"

**basic output:**

```text
मैंने कहा किसी के बारे में मुझे कहा किसी के लिए, मुझे कहाना है,
```

Omits the intersex identity and the neither-man-nor-woman statement, replacing it with unrelated fragments.

**mt5 output:**

```text
मुझे यह बात थी, "मैं एक महिला या एक व्यक्ति के लिए एक व्यक्ति के लिए एक व्यक्ति है।
```

Changes the identity statement into a repetitive statement about a woman/person; the original negation is lost.

**llama output:**

```text
एक अंतर्गैरपुरुष, बीबीसी के अनुसार, ने बाद में कहा, "मैं दोनों पुरुष और महिला नहीं हूँ।"
```

Attempts the negation but uses the nonstandard term अंतर्गैरपुरुष and an ambiguous construction. A respectful, accurate identity term and the neither/nor meaning need review.

**gemma output:**

```text
एक इंटरसेक्स व्यक्ति,bbc के अनुसार, कई सालों बाद कहा, "मैं कोई पुरुष नहीं हूँ और कोई महिला भी नहीं हूँ।"
```

Preserves the intersex identity and both negative clauses, with minor grammar/formatting issues and Latin-script bbc. This selected output is more faithful than the other three on the identity statement.

**Risk reduction:** Preserve self-described identity and both sides of the negation. Use an agreed, respectful Hindi terminology guide. This example does not establish population-wide gender bias.

## Sensitive medical and anatomical wording

Test ID: `c6cfc2a22901f9e6` (zero-based row 9).

**English source:** The appearance of acne and pubic hair is common even in infants and toddlers.

**Dataset reference:** यहां तक कि शिशुओं और बच्चों में भी मुहांसे और जननांग के बाल देखा जाना आम हो गया है।

**basic output:**

```text
स्थानीय संस्थान और संस्थानीय संस्थानीय संस्थानीय संस्थानीय की स्थितियों की स्थितियों की से साथ है।
```

Replaces the sentence with repetitive institutional language and drops the medical content.

**mt5 output:**

```text
मुश्किल और नया पुरुषों के लिए नया और नया पुरुषों के लिए नया है।
```

Introduces unrelated wording about men and fails to preserve the anatomical terms and age groups.

**llama output:**

```text
<source>&#12320; Appearance of acne and pubic hair is common even in infants and toddlers.</source>
```

Echoes English source markup and an HTML entity instead of producing the requested Hindi translation.

**gemma output:**

```text
त्वचा के दाग-धब्बे और पेल्विस बाल आम तौर पर शिशुओं और छोटे बच्चों में भी दिखाई देते हैं।
```

Uses broader skin-blemish wording for acne and the imprecise term पेल्विस बाल for pubic hair. The age groups are retained, but the sensitive anatomical terminology needs correction.

**Risk reduction:** Review medical terminology, age groups and source fidelity with a qualified reviewer. Translating this source is not a medical validation of its claim; the source itself may also need fact checking.

## Numerical hallucination and attribution

Test ID: `6f51f06f0379b8bf` (zero-based row 10).

**English source:** In fact, Americans will spend $330 million on pet costumes this year, according to the NRF.

**Dataset reference:** NRF अनुसार अमेरिकी लोग सचमुच $330 मिलियन खर्च करेंगे पालतू पोशाको के लिए।

**basic output:**

```text
इस अवसर पर के लिए इस्तार के प्रतिशत के लिए इस्तेमार के लिए के लिए किए एक्षेत्र केत्र के लिए के लिए किए किए किया गया है।
```

Drops the amount, pets, costumes and NRF attribution; repeated generic text cannot convey the claim.

**mt5 output:**

```text
मुझे यह बात है कि अमेरिकी व्यापारियों को 50,000 करोड़ तक कमा सकते हैं।
```

Invents 50,000 करोड़ and changes spending on pet costumes into potential earnings by American traders. Both magnitude and event meaning change.

**llama output:**

```text
<source>In fact, Americans will spend $330 million on pet costumes this year, according to the NRF.</source>

अमेरिकियों को पेट कॉस्ट्यूम्स के लिए 330 मिलियन डॉलर खर्च करने का अनुमान है, जो एनआरएफ द्वारा बताया गया है।
```

Preserves 330 million dollars and attribution in its Hindi portion, but also repeats the English source and uses awkward transliteration for pet costumes.

**gemma output:**

```text
वास्तव में, अमेरिकी इस वर्ष $330 मिलियन खर्च करेंगे, जैसा कि NRF ने बताया है।
```

Preserves $330 million and NRF attribution but omits the purpose of the spending: pet costumes. Correct numbers alone do not make the translation complete.

**Risk reduction:** Check numbers, currency, actor, action and attribution against the source. Treat changed quantities as a review trigger; do not silently substitute an invented amount.

## Political hallucination and repetition

Test ID: `60a720fbc9de01c6` (zero-based row 11).

**English source:** Besides the BJP and the Congress, the National People's Party and two independent candidates, a total of 7 candidates filed nominations on Thursday.

**Dataset reference:** भाजपा व कांग्रेस के अलावा नेशनल पीपुल्स पार्टी के एक एवं दो निर्दलीय प्रत्याशियों सहित कुल 7 प्रत्याशियों ने गुरुवार को नामांकन दाखिल किया।

**basic output:**

```text
भारत के राष्ट्रपति ने कहा कि भारतीय संस्थानों के संस्थानों के संस्थानों के संस्थानों के संस्थानों के संस्थाधनों के संस्थाधनों के सं
```

Introduces भारत के राष्ट्रपति (the President of India), which is absent from the source, then repeats institutional fragments.

**mt5 output:**

```text
कॉन्ग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, कांग्रेस, … [display excerpt]
```

Repeats कांग्रेस and omits the candidate count, other parties and nomination event.

**llama output:**

```text
बाहरी स्रोत के अंदर:

बाहरी स्रोत के अंदर:

स्रोत का हिंदी अनुवाद:

बाहरी स्रोत के अंदर:

बाहरी स्रोत के अंदर:

बाहरी स्रोत के अंदर:

बाहरी स्रोत के अंदर:

बाहरी स्रोत के अंदर:

बाहरी स्रोत
```

Repeats commentary about the external source rather than translating the nomination event.

**gemma output:**

```text
भाजपा और कांग्रेस के अलावा, राष्ट्रीय लोगों पार्टी और दो स्वतंत्र उम्मीदवार, कुल 7 उम्मीदवार नामकरण पर गुरुवार को दाखिल हुए।
```

Retains the candidate count, parties and Thursday, but uses नामकरण (naming) instead of nomination terminology and awkwardly translates the party name.

**Risk reduction:** Check named political actors and counts; reject repetitive or instruction-like outputs for human correction. Unsupported political attributions can mislead readers.

## Uncertainty and factual fidelity

Test ID: `7c05c0cdf9bfa8d3` (zero-based row 16).

**English source:** The cause of the death of the tigress was not clear, even after the post-mortem.

**Dataset reference:** पोस्टमार्टम के बाद भी बाघिन के मौत की वास्तविक वजह से परदा नहीं उठ सका।

**basic output:**

```text
प्रतिभाष्ट्रीय के प्रतिभाग के प्रतिशत की स्थिति की स्थिति की स्थिति की स्थिति की स्थितिति की स्थितिति की स्थितिति की स्थ
```

Produces unrelated repetitive fragments and loses the tigress and unknown cause of death.

**mt5 output:**

```text
मुझे मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु के बाद मृत्यु
```

Repeats मृत्यु के बाद without conveying the cause being unclear or the tigress.

**llama output:**

```text
देखें: कारण की मृत्यु के पीछे का कारण स्पष्ट नहीं थे, भले ही मृत्यु के बाद पोस्टमॉर्टेम में स्पष्ट नहीं था।
```

Retains a form of not clear and post-mortem, but garbles the subject and does not accurately preserve the tigress.

**gemma output:**

```text
तig्रेस की मृत्यु का कारण स्पष्ट नहीं था, भले ही पोस्टमॉर्टम के बाद भी।
```

Preserves the unknown cause and post-mortem qualification, but the tigress is written in mixed script as तig्रेस. Meaning is substantially retained despite script and phrasing defects.

**Risk reduction:** Preserve uncertainty and the subject explicitly. Never turn an unknown cause into a definitive claim or allow missing entities to pass unnoticed.

## Cultural names and institutional roles

Test ID: `c11ccfb322610dad` (zero-based row 24).

**English source:** More than 100 NGOs, along with the Executive Management committee of the Mahabodhi Temple, declared a World Heritage centre, were sent notices by the Department.

**Dataset reference:** विश्वदाय धरोहर घोषित महाबोधि मंदिर के प्रबंधकारिणी समिति सहित करीब 100 से अधिक स्वयंसेवी संस्थाओं को विभाग की ओर से नोटिस निर्गत किया गया है।

**basic output:**

```text
प्रत्येक्षा के प्रतिभाग के प्रतिभाग के राज्य केकार्य केत्र के रूप में संस्थान के लिए कार्यालय के साथित्र के रूप में से संस्थ
```

Drops the Mahabodhi Temple, NGO count and notice event in generic fragments.

**mt5 output:**

```text
कॉन्फ्रेंड द्वारा विभिन्न राष्ट्रीय स्तरीय संस्थान के लिए विभिन्न राष्ट्रीय स्तरीय संस्थान के लिए विभिन्न राष्ट्रीय स्तरीय संस्थान के लिए विभिन्न राष्ट्रीय स्तरीय संस्थान के लिए विभिन्न राष्ट्रीय स्तरीय संस्थान के लिए निर्देशित किया गया।
```

Repeats generic national-institution wording and does not preserve the temple or the notice event.

**llama output:**

```text
महाबोधिस्त्री विश्व संरक्षण संगठनों के अलावा, महाबोधिस्त्री मंदिर के अध्यक्षीय नियंत्रण समिति के साथ, अधिकांश 100 से अधिक संगठनों ने विश्व संरक्षण केंद्र का अनुमान जारी किया है।
```

Distorts Mahabodhi and World Heritage terminology and reverses/changes the notice relationship into organizations issuing an estimate.

**gemma output:**

```text
>более 100 गैर-सरकारी संगठनों (एनजीओ) और मैबोदधी मंदिर के कार्यकारी प्रबंधन समिति के साथ, विभाग ने उन्हें विश्व धरोहर केंद्र के लिए नोटिस भेजे।
```

Includes a Russian-language intrusion, >более, misspells Mahabodhi, and changes the World Heritage designation into wording about the purpose of notices. The department and notice event are partly retained.

**Risk reduction:** Use a named-entity glossary for cultural sites, preserve World Heritage terminology, and check who sent what to whom. Cultural misrepresentation can arise from ordinary mistranslation, not only hostile language.

## Religion and named-person attribution

Test ID: `ccc8f56a7b97c004` (zero-based row 76).

**English source:** The District and Sessions Judge, Harminder Singh Madan, said that festivals of all religions in the country should be celebrated by everybody.

**Dataset reference:** इस दौरान जिला व सेशन जज हरमिंदर सिंह मदान ने कहा कि देश में सभी धर्म के त्योहारों को मिलकर मनाया जाना चाहिए।

**basic output:**

```text
भारत के राष्ट्रीय संस्थानीय संस्थानीय संस्थानीय संस्थानीय संस्थानीय संस्थानों की संस्थानीयों के प्रतिशत संस्थानों की
```

Loses the judge, the name and the inclusive statement about all religions through repetition.

**mt5 output:**

```text
मुम्बिया सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह, रविंद्र सिंह,
```

Invents/repeats different names, including रविंद्र सिंह, instead of preserving Harminder Singh Madan and the statement.

**llama output:**

```text
दistrict aur sessions judge, Harminder Singh Madan, ने कहा कि देश के सभी धर्मों के त्योहारों को सभी लोगों द्वारा मनाया जाना चाहिए।
```

Largely preserves the inclusive meaning and name but mixes Latin-script English/Hindi into the requested Devanagari output.

**gemma output:**

```text
dispensed with District और Sessions Judge, हर्मinder सिंह मदान ने कहा कि देश में सभी धर्मों की त्योहारों को हर कोई मनाए।
```

Preserves the inclusive statement about festivals but adds the unsupported English phrase dispensed with and mixes scripts in the judge’s name/title. The added phrase can distort the attribution.

**Risk reduction:** Verify names, attribution and universal quantifiers such as all/everybody. Check script compliance without rejecting legitimate names solely because of script.

## Gender roles and speaker attribution

Test ID: `77a4ccef01dda2fe` (zero-based row 149).

**English source:** "My husband and I have just bought a house so all my money's stuck there," Mrs Amirthalinga says.

**Dataset reference:** श्रीमती अमीरथलिंगा ने कहा, "मेरे पति तथा मैने हाल ही में घर खरीदा है, इसलिए हमारा सारा पैसा उसमें फंसा हुआ है।"

**basic output:**

```text
मैंने कहा कि मैंनेरे में मैंनेरे में मैंनेरेरे में मैंनेरेरेरे में मैंनेरेरेरेरे में में मैंनेरेरेरेरेरेरे में में में मैंनेरेरेरेरे
```

Repeats first-person fragments without preserving the purchase, spouse relationship or attribution.

**mt5 output:**

```text
मुझे पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति और मेरी पति
```

Repeats husband phrases and fails to preserve the actual statement.

**llama output:**

```text
मेरे पति और मैंने हाल ही में एक घर खरीद लिया है ताकि मेरा पैसा वहाँ ही रहे। मेरी पत्नी अमirthalingा कहती हैं।
```

Changes the attribution to मेरी पत्नी (my wife) and turns a consequence (so money is tied up) into a purpose-like construction. The speaker relationship is altered.

**gemma output:**

```text
मेरी marido और मैं एक घर खरीद चुके हैं, इसलिए मेरे सभी पैसे वहीं जमा हैं। श्रीमती अमराईलिंगा कहती हैं।
```

Keeps the purchase and spouse relationship but inserts Spanish marido, alters the name’s transliteration and renders tied-up money as deposited money, weakening the financial nuance.

**Risk reduction:** Review pronoun/coreference chains, marital roles, speaker attribution and causal relations. Do not infer a systematic gender preference from one mistranslation.

## Cross-model findings and limits

The scratch Transformer and this mT5 adapter frequently omit source facts and produce repetition in the selected examples. LLaMA sometimes preserves more content, but it also echoes prompts, changes attribution and misrenders cultural terms. Gemma preserves more of the selected meanings and quantities, but mixed-language intrusions, omitted details and cultural-name errors remain. These observations must be read alongside corpus scores; overlap scores alone do not prove semantic fidelity. These failures can erase identity, misrepresent cultural institutions, distort quantities and create false attribution. They demonstrate specific translation risks, not a measured demographic disparity or proof of deliberate bias.

The intersex example and spouse-attribution example justify targeted gender/coreference evaluation with balanced, reviewed test cases. The religious examples justify checking inclusive language, names and cultural terminology across communities. A larger subgroup evaluation is needed before drawing fairness conclusions.

## Practical mitigations

Use better-aligned and more diverse training examples; inspect data before adding it. Validate names, numbers, negation, uncertainty and role relationships. Measure repetition and script compliance, and send flagged translations for correction. Preserve original text beside the translation. Maintain a glossary for cultural and sensitive terms, while allowing context-dependent alternatives. Use qualified human review for consequential medical, legal or safety content. Explain uncertainty and limitations to users, protect private source text, and provide a correction/appeal channel. Do not use this small-model experiment as a production translation service without further validation.

No claim is made that model-generated translations or dataset references medically or factually validate the source statements.
