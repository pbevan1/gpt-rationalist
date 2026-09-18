# Style references, not training text

The user's target is verbal mannerisms, entirely independent of subject matter.
The training dataset is therefore **project-original synthetic everyday answers**.
The collected articles are a reference for word choice and register, never their
training targets. `prepare.py` reads only `style_pairs.jsonl`, not `data/raw/`.

| Source | Public collection route | Local posts |
| --- | --- | ---: |
| [LessWrong](https://www.lesswrong.com/) | Public GraphQL, top posts, minimum score 50 | 1,200 |
| [EA Forum](https://forum.effectivealtruism.org/) | Public GraphQL, top posts, minimum score 30 | 800 |
| [Alignment Forum](https://www.alignmentforum.org/) | Public GraphQL, top posts, minimum score 20 | 250 |
| [Don't Worry About the Vase](https://thezvi.wordpress.com/) | Public WordPress.com API, recent posts | 200 |
| [Slate Star Codex](https://slatestarcodex.com/) | Public WordPress REST API, recent archived posts | 200 |

Snapshots were fetched on 2026-09-16. The reference collection is deliberately a
small sample, not a full historical archive. The forums contain cross-posts, so
2,650 is a record count, not a count of unique works. Phrase counts from
`reference.py` are exploratory and may include quotes, titles, and repeated posts.

Each raw record keeps title, author, source URL, date, source ID, retrieval time,
and a rights note. Original posts remain their authors' material; no blanket corpus
license is asserted. The cloud bundle excludes these posts and extracted fragments.
The model's [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE)
is separate from source-post rights.

The reference APIs were tested live. The [forum API tutorial](https://www.lesswrong.com/posts/LJiGhpq8w4Badr5KJ/graphql-tutorial-for-lesswrong-and-effective-altruism-forum)
and [ForumMagnum repository](https://github.com/ForumMagnum/ForumMagnum) document the
forum implementation. Zvi's [about page](https://thezvi.substack.com/about) identifies
the ongoing WordPress mirror. [WordPress.com API docs](https://developer.wordpress.com/docs/api/)
and [WordPress REST posts docs](https://developer.wordpress.org/rest-api/reference/posts/)
describe the two blog collection routes.

The vocabulary expansion was checked against these local references. For example,
the collection contains “orthogonal”, “load-bearing”, “directionally”, “legible”,
“on the margin”, “cached thought”, and “gears-level”. That verifies usage in the
sample, not exclusivity to this community. `reference.py` can now extract short
attributed contexts for the broader vocabulary too. None of those excerpts are
used as training answers.

There is no source text in `style_pairs.jsonl`. Every row explicitly pairs a plain
answer with an original stylised response, with occasional playful metaphors.
Those pairs are not claims about any individual author's views or endorsements.
Use the plain answers to review semantic drift when editing or expanding the data.
