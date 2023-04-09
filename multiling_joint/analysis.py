import os
import json

dirs = os.listdir('result/')
        
average_intent = 0
average_f1_score = 0
average_exact_match = 0
best_exact_match = 0
best_intent = 0
best_f1 = 0
best_exact_match_lang = ''
worst_exact_match = 100
worst_exact_match_lang = ''
worst_intent = 0
worst_f1 = 0
for dir_ in dirs:
    with open(f'result/{dir_}/eval_metric.json', 'r') as f:
        file = json.load(f)
        intent = file['intent_acc']
        f1 = file['slot_f1']
        match = file['sementic_frame_acc']

        average_intent += intent
        average_f1_score += f1
        average_exact_match += match

        if match > best_exact_match:
            best_exact_match = match
            best_intent = intent
            best_f1 = f1
            best_exact_match_lang = dir_
        if worst_exact_match > match:
            worst_exact_match = match
            worst_intent = intent
            worst_f1 = f1
            worst_exact_match_lang = dir_

average_intent /= len(dirs)
average_f1_score /= len(dirs)
average_exact_match /= len(dirs)
print(f"Average intent: {average_intent}, f1-score: {average_f1_score}, exact_match: {average_exact_match}")
print(f"Lang: {worst_exact_match_lang},  Worst intent: {worst_intent}, f1-score: {worst_f1}, exact_match: {worst_exact_match}")
print(f"Lang: {best_exact_match_lang}, Best intent: {best_intent}, f1-score: {best_f1}, exact_match: {best_exact_match}")