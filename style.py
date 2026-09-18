"""Observable verbal tics, used for passage selection and descriptive evaluation."""
import re

PATTERNS = {
    "epistemic": r"\bepistemic(?:ally)?\b",
    "excited": r"\bexcited\b",
    "capitalized_phrase": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,}\b",
    "my_model": r"\b(?:my (?:(?:current|tentative) )?model|my best guess)\b",
    "priors_updates": r"\b(?:priors?|update towards|update away|updating on|update on|bayesian|bayes)\b",
    "hedges": r"\b(?:on the margin|at the margin|plausibly|directionally|I notice I am confused|I notice that I am confused|to be clear|to be explicit|something like)\b",
    "community_terms": r"\b(?:inside view|outside view|object.level|meta.level|cached thoughts?|steelmanning|steelman|coordination problem|inferential distance|revealed preference|expected value|counterfactual|order of magnitude|low.hanging fruit)\b",
    "orthogonal": r"\borthogonal(?:ly)?\b",
    "crux": r"\b(?:crux|cruxes)\b",
    "load_bearing": r"\bload[ -]bearing\b",
    "directionally": r"\bdirectionally\b",
    "operationalise": r"\boperationali[sz](?:e|ed|ing|ation)\b",
    "legibility": r"\b(?:legible|illegible|legibility)\b",
    "on_the_margin": r"\b(?:on|at) the margin\b",
    "nontrivial": r"\bnon[ -]?trivial(?:ly)?\b",
    "bottleneck": r"\bbottlenecks?\b",
    "conditional": r"\b(?:conditional on|holding (?:\w+ )?fixed|all else equal)\b",
    "cached_thought": r"\bcached thoughts?\b",
    "views_levels": r"\b(?:inside view|outside view|object[ -]level|meta[ -]level)\b",
    "charitable_disagreement": r"\b(?:steelman(?:ning)?|sympathetic to|strongest version|push back)\b",
    "counterfactual": r"\bcounterfactual(?:ly)?\b",
    "gears_level": r"\bgears[ -]level\b",
    "affordance": r"\baffordances?\b",
    "salient": r"\b(?:salient|salience)\b",
    "robust_actionable": r"\b(?:robust(?:ly)?|actionable)\b",
    "scope_qualifiers": r"\b(?:locally|insofar as|to a first approximation|in the limit|on the relevant margin)\b",
}


def markers(text):
    return {name: len(re.findall(pattern, text, flags=0 if name == "capitalized_phrase" else re.I))
            for name, pattern in PATTERNS.items()}


def score(text):
    counts = markers(text)
    return sum(min(value, 4) * (3 if key in ("epistemic", "excited", "capitalized_phrase") else 1)
               for key, value in counts.items())
