"""Assemble the documented close reading with verbatim generated evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project.common import read_json, digest

# These observations were written after reading the saved outputs; they are not an automatic classifier.
NOTES = {
    5: {'category': 'Gender identity and negation',
        'basic': 'Omits the intersex identity and the neither-man-nor-woman statement, replacing it with unrelated fragments.',
        'mt5': 'Changes the identity statement into a repetitive statement about a woman/person; the original negation is lost.',
        'llama': 'Attempts the negation but uses the nonstandard term अंतर्गैरपुरुष and an ambiguous construction. A respectful, accurate identity term and the neither/nor meaning need review.',
        'mitigation': 'Preserve self-described identity and both sides of the negation. Use an agreed, respectful Hindi terminology guide. This example does not establish population-wide gender bias.'},
    9: {'category': 'Sensitive medical and anatomical wording',
        'basic': 'Replaces the sentence with repetitive institutional language and drops the medical content.',
        'mt5': 'Introduces unrelated wording about men and fails to preserve the anatomical terms and age groups.',
        'llama': 'Echoes English source markup and an HTML entity instead of producing the requested Hindi translation.',
        'mitigation': 'Review medical terminology, age groups and source fidelity with a qualified reviewer. Translating this source is not a medical validation of its claim; the source itself may also need fact checking.'},
    10: {'category': 'Numerical hallucination and attribution',
        'basic': 'Drops the amount, pets, costumes and NRF attribution; repeated generic text cannot convey the claim.',
        'mt5': 'Invents 50,000 करोड़ and changes spending on pet costumes into potential earnings by American traders. Both magnitude and event meaning change.',
        'llama': 'Preserves 330 million dollars and attribution in its Hindi portion, but also repeats the English source and uses awkward transliteration for pet costumes.',
        'mitigation': 'Check numbers, currency, actor, action and attribution against the source. Treat changed quantities as a review trigger; do not silently substitute an invented amount.'},
    11: {'category': 'Political hallucination and repetition',
        'basic': 'Introduces भारत के राष्ट्रपति (the President of India), which is absent from the source, then repeats institutional fragments.',
        'mt5': 'Repeats कांग्रेस and omits the candidate count, other parties and nomination event.',
        'llama': 'Repeats commentary about the external source rather than translating the nomination event.',
        'mitigation': 'Check named political actors and counts; reject repetitive or instruction-like outputs for human correction. Unsupported political attributions can mislead readers.'},
    16: {'category': 'Uncertainty and factual fidelity',
        'basic': 'Produces unrelated repetitive fragments and loses the tigress and unknown cause of death.',
        'mt5': 'Repeats मृत्यु के बाद without conveying the cause being unclear or the tigress.',
        'llama': 'Retains a form of not clear and post-mortem, but garbles the subject and does not accurately preserve the tigress.',
        'mitigation': 'Preserve uncertainty and the subject explicitly. Never turn an unknown cause into a definitive claim or allow missing entities to pass unnoticed.'},
    24: {'category': 'Cultural names and institutional roles',
        'basic': 'Drops the Mahabodhi Temple, NGO count and notice event in generic fragments.',
        'mt5': 'Repeats generic national-institution wording and does not preserve the temple or the notice event.',
        'llama': 'Distorts Mahabodhi and World Heritage terminology and reverses/changes the notice relationship into organizations issuing an estimate.',
        'mitigation': 'Use a named-entity glossary for cultural sites, preserve World Heritage terminology, and check who sent what to whom. Cultural misrepresentation can arise from ordinary mistranslation, not only hostile language.'},
    76: {'category': 'Religion and named-person attribution',
        'basic': 'Loses the judge, the name and the inclusive statement about all religions through repetition.',
        'mt5': 'Invents/repeats different names, including रविंद्र सिंह, instead of preserving Harminder Singh Madan and the statement.',
        'llama': 'Largely preserves the inclusive meaning and name but mixes Latin-script English/Hindi into the requested Devanagari output.',
        'mitigation': 'Verify names, attribution and universal quantifiers such as all/everybody. Check script compliance without rejecting legitimate names solely because of script.'},
    149: {'category': 'Gender roles and speaker attribution',
        'basic': 'Repeats first-person fragments without preserving the purchase, spouse relationship or attribution.',
        'mt5': 'Repeats husband phrases and fails to preserve the actual statement.',
        'llama': 'Changes the attribution to मेरी पत्नी (my wife) and turns a consequence (so money is tied up) into a purpose-like construction. The speaker relationship is altered.',
        'mitigation': 'Review pronoun/coreference chains, marital roles, speaker attribution and causal relations. Do not infer a systematic gender preference from one mistranslation.'},
}


def build(run):
    run = Path(run)
    models = {name: read_json(run / 'predictions' / f'{name}.json') for name in ('basic', 'mt5', 'llama', 'gemma')}
    evidence = read_json(run / 'results/review_evidence.json')
    for name, rows in models.items():
        if digest(rows) != evidence['prediction_hashes'][name]:
            raise ValueError(f'{name} outputs changed: re-read them and revise the qualitative observations first.')
    gemma_notes = read_json(run / 'results/gemma_review_notes.json')
    lines = ['# Qualitative translation and responsible AI review', '',
        'This is an AI-assisted close reading of eight deliberately selected test examples shared by all four models. '
        'It is not an independent professional translator assessment, a random audit, or a statistical fairness study. '
        'Examples were selected for identity, religion, cultural names, numerical claims, uncertainty and sensitive content. '
        'Claims below concern the displayed outputs only; no model is declared unbiased or safe from these examples.', '',
        'Verbatim output excerpts are limited to 240 characters for readability. The complete outputs remain in the prediction JSON files; '
        'an ellipsis added here marks display shortening, not a change to the evaluated text.']
    for i, notes in NOTES.items():
        row = models['basic'][i]
        lines += ['', f'## {notes["category"]}', '', f'Test ID: `{row["id"]}` (zero-based row {i}).', '',
                  '**English source:** ' + row['en'], '', '**Dataset reference:** ' + row['hi'], '']
        for name, rows in models.items():
            assert rows[i]['id'] == row['id']
            output = rows[i]['prediction']
            excerpt = output[:240] + (' … [display excerpt]' if len(output) > 240 else '')
            note = gemma_notes[str(i)] if name == 'gemma' else notes[name]
            lines += [f'**{name} output:**', '', '```text', excerpt, '```', '', note, '']
        lines += ['**Risk reduction:** ' + notes['mitigation']]
    lines += ['', '## Cross-model findings and limits', '',
        'The scratch Transformer and this mT5 adapter frequently omit source facts and produce repetition in the selected examples. '
        'LLaMA sometimes preserves more content, but it also echoes prompts, changes attribution and misrenders cultural terms. '
        'Gemma preserves more of the selected meanings and quantities, but mixed-language intrusions, omitted details and cultural-name errors remain. '
        'These observations must be read alongside corpus scores; overlap scores alone do not prove semantic fidelity. '
        'These failures can erase identity, misrepresent cultural institutions, distort quantities and create false attribution. '
        'They demonstrate specific translation risks, not a measured demographic disparity or proof of deliberate bias.', '',
        'The intersex example and spouse-attribution example justify targeted gender/coreference evaluation with balanced, reviewed test cases. '
        'The religious examples justify checking inclusive language, names and cultural terminology across communities. '
        'A larger subgroup evaluation is needed before drawing fairness conclusions.', '',
        'A separate reference-quality check found weak alignment at test ID `338385764865c774`: the English source concerns cleaning a park, '
        'but its Hindi reference discusses correcting the poor condition of departments. This is a dataset limitation, not a model-generated error. '
        'Automatic overlap scores on such cases need cautious interpretation.', '',
        '## Practical mitigations', '',
        'Use better-aligned and more diverse training examples; inspect data before adding it. Validate names, numbers, negation, uncertainty and role relationships. '
        'Measure repetition and script compliance, and send flagged translations for correction. Preserve original text beside the translation. '
        'Maintain a glossary for cultural and sensitive terms, while allowing context-dependent alternatives. Use qualified human review for consequential medical, legal or safety content. '
        'Explain uncertainty and limitations to users, protect private source text, and provide a correction/appeal channel. '
        'Do not use this small-model experiment as a production translation service without further validation.', '',
        'No claim is made that model-generated translations or dataset references medically or factually validate the source statements.']
    (run / 'results/QUALITATIVE_REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(run / 'results/QUALITATIVE_REVIEW.md')


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'runs/laptop')
