import re
import pandas as pd

def identify_parents(df, text_column='post'):

    mother_markers = [
        r'\bi\s*(\w+\s)*?([\’\']?m|\sam|was|have\sbeen|have\sto\be)\s*(a|the)?\s*(\w+\s)*?(m[uo]m|mother|stahm)\b',
        r'becoming\s*(a|the)\s*(m[uo]m|mother|stahm)',
        r'i\s*(\w+\s)*?([\’\']?m|\sam|was|have\sbeen)\s*(\w+\s)*?\d+\s*months\spostpartum',
        r'\b(their|his|her|child[\’\']?s|baby[\’\']s|son[\’\']?s|daughter[\’\']?s)\s*(\w+\s)*?(dad|father)',
        r'((\bf\s?[\/\\]?\s?\d+\b)|(\b\d+\s?[\/\\]?\s?f\b))',
        r'i?\s*(has|got)?\s*(ppd|postpartum\sdepression)',
        r'mother\sto\b',
        r'to\s*be\s*(a|the)?\s*(mother|m[ou]m)',
        r'my\s*\.*?\s*(husband|boyfriend|fiance|bf)',
        r'i\s*(\w+\s)*?([\’\']?m|am|was|got)\s*pregnant',
        r'my\s*pregnancy',
        r'i\s*gave\s*birth',
        r'i\s*delivered\s*(a|the)?\s*baby',
        r'(as|being)\s*(a|the)?\s*(woman|m[uo]m|mother)',
        r'\bi\s*.*?had\s*(a|my)?\s*(\w+\s)*?(son|daughter|baby)',
        r'i\s*(became|have\sbecome)\s*(a|the)?\s*m(other|om(my)?)',
        r'i\s*hate\s*motherhood',
        r'i\s*hate\s*being\s*(a|the)?\s*(m[uo]m|mother|stahm|wife)',
        r'baby\s*daddy',
        r'(my)?\s*(he|husband|boyfriend|fiance|bf)\s*(\w+\s)*?want(s|ed)\s*(kids?|child(ren)?)'
    ]

    father_markers = [
        r'i\s*([\’\']?m|am)\s*(a|the)?\s*(\w+\s)*?(dad|father|stahd)',
        r'becoming\s*(a|the)\s*(dad|father|stahd)',
        r'\b(their|his|her|child[\’\']?s|baby[\’\']?s|son[\’\']?s|daughter[\’\']?s)\s*(m[uo]m|mother)',
        r'((\bm\s?[\/\\]?\s?\d+\b)|(\b\d+\s?[\/\\]?\s?m\b))',
        r'father\s*to\b',
        r'my\s*(\w+\s)*?\s*\b(wife|girlfriend|gf)',
        r'(my)?\s*(wife|girlfriend|gf|partner)[\’\']?s\s*(\w+\s)*?(pregnancy|ppd|postpartum\sdepression)',
        r'(my)?\s*(she|wife|girlfriend|fiance|gf|partner)\s*([\’\']?s|is|was|got)\s*preg(nant)?',
        r'(my)?\s*(she|wife|girlfriend|fiance|gf)\s*(has|got)?\s*(ppd|postpartum depression)',
        r'(my)?\s*(she|wife|girlfriend|fiance|gf)\s*(\w+\s)*?want(s|ed)\s*(kids?|child(ren)?)',
        r'(as|being)\s*(a|the)\s*(man|dad|father)',
        r'i\s*hate\s*being\s*(a|the)?\s*(father|dad|husband)',
        r'i\s*hate\s*fatherhood',
        r'to\s*be\s*(a|the)?\s*(father|dad)'
    ]

    def count_markers(text, markers):
        text_lower = text.lower()
        count = 0
        for marker in markers:
            matches = len(re.findall(marker, text_lower))
            count += matches
        return count

    def determine_parent(text):
        mother_count = count_markers(text, mother_markers)
        father_count = count_markers(text, father_markers)

        if mother_count > father_count:
            return f'Likely mother: {mother_count} > {father_count}'
        elif father_count > mother_count:
            return f'Likely father: {father_count} > {mother_count}'
        else:
            return f'Unclear: {mother_count} / {father_count}'
    
    df['parent'] = df[text_column].apply(determine_parent)

    return df

df = pd.read_csv('regretful_parents_posts.csv')
df = identify_parents(df, text_column='post')
df.to_csv('regretful_parents_posts_gendered.csv')

mothers = len(df[df['parent'].str.contains('Likely mother', na=False)])
fathers = len(df[df['parent'].str.contains('Likely father', na=False)])

print(f"Likely mothers: {mothers}")
print(f"Likely fathers: {fathers}")
