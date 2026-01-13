import pandas as pd
import random
import re
from tqdm import tqdm

random.seed(42)


def augment_text_synonym(text):
    replacements = {
        'bad': ['terrible', 'awful', 'poor', 'horrible'],
        'terrible': ['bad', 'awful', 'horrible', 'dreadful'],
        'good': ['great', 'nice', 'excellent', 'wonderful'],
        'hate': ['dislike', 'despise', 'detest', 'loathe'],
        'love': ['like', 'enjoy', 'adore', 'appreciate'],
        'worst': ['poorest', 'terrible', 'awful', 'horrible'],
        'best': ['greatest', 'finest', 'excellent', 'top'],
        'amazing': ['incredible', 'awesome', 'fantastic', 'wonderful'],
        'stupid': ['dumb', 'foolish', 'idiotic', 'silly'],
        'great': ['excellent', 'wonderful', 'fantastic', 'amazing'],
        'not': ['never', 'hardly', 'barely', 'rarely'],
        'really': ['very', 'extremely', 'incredibly', 'truly'],
        'never': ['not', 'hardly', 'rarely', 'seldom'],
    }
    
    words = text.split()
    if len(words) < 3:
        return text
    
    num_replacements = min(2, len(words) // 5 + 1)
    
    for _ in range(num_replacements):
        idx = random.randint(0, len(words) - 1)
        word = words[idx].lower().strip('.,!?')
        
        if word in replacements:
            synonym = random.choice(replacements[word])
            if words[idx][0].isupper():
                synonym = synonym.capitalize()
            
            punct = ''.join([c for c in words[idx] if c in '.,!?'])
            words[idx] = synonym + punct
    
    return ' '.join(words)


def augment_text_add_phrase(text, sentiment):
    positive_prefixes = ["Wow! ", "Great! ", "Amazing! ", "Excellent! ", "Awesome! "]
    positive_suffixes = [" Love it!", " So good!", " Highly recommend!", " Best ever!", " Perfect!"]
    
    negative_prefixes = ["Ugh! ", "Terrible! ", "Awful! ", "Horrible! ", "Disappointed! "]
    negative_suffixes = [" Hate it!", " So bad!", " Never again!", " Worst ever!", " Avoid!"]
    
    neutral_prefixes = ["Well, ", "Actually, ", "Honestly, ", "Basically, ", "Technically, "]
    neutral_suffixes = [" Not sure.", " Maybe.", " Could be better.", " It's okay.", " I guess."]
    
    choice = random.choice(['prefix', 'suffix', 'none'])
    
    if choice == 'prefix':
        if sentiment == 'positive':
            return random.choice(positive_prefixes) + text
        elif sentiment == 'negative':
            return random.choice(negative_prefixes) + text
        else:
            return random.choice(neutral_prefixes) + text
    
    elif choice == 'suffix':
        if sentiment == 'positive':
            return text + random.choice(positive_suffixes)
        elif sentiment == 'negative':
            return text + random.choice(negative_suffixes)
        else:
            return text + random.choice(neutral_suffixes)
    
    return text


def augment_text_paraphrase(text):
    patterns = [
        (r'\bI think\b', random.choice(['I believe', 'I feel', 'In my opinion', 'I guess'])),
        (r'\bThis is\b', random.choice(['It is', 'This seems', 'It appears', 'It looks like'])),
        (r'\bvery\b', random.choice(['really', 'extremely', 'quite', 'super'])),
        (r'\bso\b', random.choice(['very', 'really', 'extremely', 'pretty'])),
        (r'\bnice\b', random.choice(['good', 'great', 'cool', 'pleasant'])),
        (r'\bbad\b', random.choice(['poor', 'terrible', 'awful', 'horrible'])),
    ]
    
    pattern, replacement = random.choice(patterns)
    text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
    
    return text


def generate_synthetic_comment(base_text, sentiment, method='synonym'):
    if method == 'synonym':
        return augment_text_synonym(base_text)
    elif method == 'phrase':
        return augment_text_add_phrase(base_text, sentiment)
    elif method == 'paraphrase':
        return augment_text_paraphrase(base_text)
    elif method == 'combo':
        text = augment_text_synonym(base_text)
        text = augment_text_add_phrase(text, sentiment)
        return text
    else:
        return base_text


def create_balanced_dataset(input_file, output_file, target_per_class=4500):
    print("\n" + "="*70)
    print("CREATING BALANCED SYNTHETIC DATASET")
    print("="*70)
    
    print(f"\nReading original dataset: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df):,} comments")
    print(f"\nOriginal distribution:")
    for sentiment, count in df['Sentiment'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment:10s}: {count:6,} ({percentage:5.2f}%)")
    
    sentiment_classes = ['negative', 'neutral', 'positive']
    balanced_data = []
    
    for sentiment in sentiment_classes:
        print(f"\nProcessing {sentiment} comments...")
        
        sentiment_df = df[df['Sentiment'] == sentiment]
        original_count = len(sentiment_df)
        
        print(f"  Original count: {original_count:,}")
        print(f"  Target count: {target_per_class:,}")
        print(f"  Need to generate: {max(0, target_per_class - original_count):,} synthetic samples")
        
        balanced_data.extend(sentiment_df.to_dict('records'))
        
        if original_count < target_per_class:
            needed = target_per_class - original_count
            
            methods = ['synonym', 'phrase', 'paraphrase', 'combo']
            samples_per_method = needed // len(methods) + 1
            
            for method in tqdm(methods, desc=f"  Generating {sentiment}"):
                for _ in range(samples_per_method):
                    if len([x for x in balanced_data if x['Sentiment'] == sentiment]) >= target_per_class:
                        break
                    
                    base_comment = random.choice(sentiment_df['Comment'].tolist())
                    synthetic_comment = generate_synthetic_comment(base_comment, sentiment, method)
                    
                    balanced_data.append({
                        'Comment': synthetic_comment,
                        'Sentiment': sentiment
                    })
            
            current_count = len([x for x in balanced_data if x['Sentiment'] == sentiment])
            print(f"  Final count: {current_count:,}")
    
    print("\n" + "="*70)
    print("CREATING FINAL DATASET")
    print("="*70)
    
    balanced_df = pd.DataFrame(balanced_data)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal dataset size: {len(balanced_df):,} comments")
    print(f"\nFinal distribution:")
    for sentiment, count in balanced_df['Sentiment'].value_counts().items():
        percentage = (count / len(balanced_df)) * 100
        print(f"  {sentiment:10s}: {count:6,} ({percentage:5.2f}%)")
    
    balanced_df.to_csv(output_file, index=False)
    print(f"\nBalanced dataset saved to: {output_file}")
    
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    input_file = "YoutubeCommentsDataSet.csv"
    output_file = "YoutubeCommentsDataSet_Balanced.csv"
    target_samples_per_class = 4500
    
    create_balanced_dataset(input_file, output_file, target_samples_per_class)
