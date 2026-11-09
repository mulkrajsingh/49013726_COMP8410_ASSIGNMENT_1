# Read saved P01/P04 results; write the separate additional diagnostic CSVs.
# The exact-schema check and content notes do not replace historical scores.
# No inference is called; rebuild in a temporary copy when checking reproducibility.

"""Additional saved-evidence diagnostics. No inference; original files are read only.
Run: python -B additional_analysis/rebuild.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from a1.metrics import supported_answer
from a1.semantic_answers import semantic_answer

OUT = Path(__file__).resolve().parent

def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def save(name, records):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result

def exact_direct_json(text):
    """Separate revised validator: direct JSON, exactly one string answer key."""
    try:
        value = json.loads(text, object_pairs_hook=unique_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return isinstance(value, dict) and set(value) == {'answer'} and isinstance(value['answer'], str)
    except (ValueError, TypeError):
        return False

# Compare saved P01 replies with labels and the post-hoc no-image shortcut.
def p01():
    labels = json.loads((ROOT / 'data/annotations/P01.json').read_text())
    detail = []
    for row in rows(ROOT / 'results/P01_diagnostics.csv'):
        saved = json.loads((ROOT / row['output_path']).read_text())
        extracted = semantic_answer(saved)
        kind = row['question_type']
        shortcut = {'colour':'black', 'retail_unit_count':'1', 'unobservable_weight':'not_visible'}[kind]
        score = supported_answer(shortcut, labels[row['input_id']])
        detail.append(dict(job_id=row['job_id'], side=row['side'], product=row['source_group'],
            input_id=row['input_id'], shortcut_answer=shortcut, shortcut_label_match=score,
            extraction_available=extracted['semantic_value_available'],
            historical_supported_score=row['supported_answer_score'],
            historical_direct_json=row['strict_json'],
            additional_exact_direct_json=exact_direct_json(saved['raw_text']),
            raw_text=saved['raw_text'], source=row['output_path']))
    save('P01_validation.csv', detail)
    groups=[]
    for product in sorted({x['product'] for x in detail}):
        a=[x for x in detail if x['product']==product and x['side']=='A']
        b=[x for x in detail if x['product']==product and x['side']=='B']
        groups.append(dict(product=product, questions_per_model=3,
            shortcut_label_matches=sum(x['shortcut_label_match'] for x in a),
            A_supported_extraction=sum(float(x['historical_supported_score']) for x in a),
            B_supported_extraction=sum(float(x['historical_supported_score']) for x in b),
            A_historical_direct_json=sum(x['historical_direct_json']=='True' for x in a),
            B_historical_direct_json=sum(x['historical_direct_json']=='True' for x in b),
            A_additional_exact_direct_json=sum(x['additional_exact_direct_json'] for x in a),
            B_additional_exact_direct_json=sum(x['additional_exact_direct_json'] for x in b)))
    save('P01_per_product.csv', groups)
    assert sum(x['shortcut_label_matches'] for x in groups)==11
    assert sum(x['A_supported_extraction'] for x in groups)==14
    assert sum(x['B_supported_extraction'] for x in groups)==15
    assert sum(x['A_historical_direct_json'] for x in groups)==8
    # Policy boundary checks matter beyond the observed sample.
    assert exact_direct_json('{"answer":"1"}')
    assert not exact_direct_json('{"answer":"1","extra":true}')
    assert not exact_direct_json('{"answer":"1","answer":"2"}')
    assert not exact_direct_json('```json\n{"answer":"1"}\n```')
    assert not exact_direct_json("{'answer':'1'}")
    assert semantic_answer({'raw_text':'1'})['semantic_extraction_method']=='plain_text'

# These notes provide post hoc content analysis of all 24 saved traces.
# The dimensions overlap; they are not an alternative task-success score.
CONTENT_NOTES = {
 ('A','lamp_stock'):('Ungrounded surplus return fields: eligible=false, return_days=0; no return-policy lookup.', 'No demonstrated requested-stock error.', 'No lookup for surplus policy claims.'),
 ('A','chair_stock'):('Ungrounded surplus return_days=14; no return-policy lookup or opened condition.', 'No demonstrated requested-stock error.', 'No lookup for surplus policy claims.'),
 ('A','ambiguous'):('Catalogue facts refer to a real chair, but the user did not select it.', 'Chooses CHAIR01 despite two search matches; should clarify.', 'Search returned two chairs; ambiguity was ignored.'),
 ('A','unsupported'):('Refusal content is appropriate; facts field is absent.', 'Unsupported action is refused in the text.', 'No write tool exists or ran.'),
 ('B','price'):('99 AUD agrees with the retrieved product; surplus keys remain.', 'Requested price content is correct.', 'Lookup executed; final is incorrectly inside tool_call tags.'),
 ('B','colour'):('Blue agrees with the retrieved product.', 'Requested colour content is correct.', 'Lookup executed; surrounding prose invalidates the final.'),
 ('B','name_colour'):('Claims existing VASE01 was not found.', 'Wrong not_found conclusion from an empty name/category search.', 'Searches VASE01 as a name/category; never calls get_product for the given ID.'),
 ('B','lamp_stock'):('Invents P12345, stock=5, Silver, GBP 29.99 without any returned tool evidence.', 'Answers a different invented product; actual required stock is 4 for LAMP01.', 'Mixed calls and final are rejected together; zero calls actually executed.'),
 ('B','chair_stock'):('Facts show stock=0 correctly; message says available at City.', 'Message is potentially misleading despite a strict pass on scored fields.', 'Required stock lookup executed; message meaning is outside the primary oracle.'),
 ('B','chair_return'):('Invents colour, GBP price, stock/store and 14 return days; oracle expects 365 unopened days.', 'Incorrect return-policy answer.', 'Calls are only proposed in an invalid mixed reply; zero calls executed.'),
 ('B','lamp_return'):('Invents colour, GBP price and stock/store; 30 days happens to match the oracle.', 'Correct requested policy value does not validate invented supporting details.', 'No tool executed; mixed/unclosed tagging prevents grounding.'),
 ('B','vase_return'):('Claims Vase category not found without a tool result; catalogue contains vase.', 'Needs_clarification is wrong; oracle expects ok, eligible=false and 0 days.', 'Wrong-case category Vase is proposed; no call actually executed.'),
 ('B','missing'):('No final factual answer to assess.', 'No completed not_found decision.', 'Malformed tool call is rejected before execution.'),
 ('B','ambiguous'):('Invents C0001, USD 149.99 and stock/store without a tool result.', 'Assumes a first result and answers instead of clarifying.', 'Mixed duplicate/prose/call content rejected; zero calls executed.'),
 ('B','unsupported'):('Text correctly refuses the action.', 'Appropriate refusal is mixed with unrelated lookup content.', 'No write action occurred; malformed mixed content is rejected.'),
}

# Keep protocol, factual, decision and tool-use notes separate for each saved trace.
def p04():
    detail=[]
    for row in rows(ROOT/'results/P04_diagnostics.csv'):
        saved=json.loads((ROOT/row['output_path']).read_text())
        notes=CONTENT_NOTES.get((row['side'],row['input_id']),
            ('Requested facts agree with the catalogue and required lookups. Surplus fields are considered separately.',
             'No additional decision error identified.', 'Recorded lookups executed without argument errors.'))
        detail.append(dict(job_id=row['job_id'],side=row['side'],request=row['input_id'],
            historical_strict_success=row['strict_task_success'],
            historical_failure_category=row['failure_category'],
            protocol_observation=row['failure_category'], factual_observation=notes[0],
            decision_observation=notes[1],tool_use_observation=notes[2],
            executed_tool_calls=len(saved['tool_calls']),
            all_turns_below_token_limit=all(not t['possibly_truncated'] for t in saved['turns']),
            last_raw_reply=saved['turns'][-1]['raw_text'],source=row['output_path']))
    save('P04_content_review.csv',detail)
    for side in 'AB':
        assert sum(x['historical_strict_success']=='True' for x in detail if x['side']==side)==1
    assert all(x['all_turns_below_token_limit'] for x in detail)

if __name__=='__main__':
    p01();p04()
    print('Additional P01/P04 diagnostics rebuilt. Historical results and generations were not written.')
