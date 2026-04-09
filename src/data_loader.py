from conllu import parse_incr


def load_conllu(file_path):

    sentences = []
    tags = []

    with open(file_path, "r", encoding="utf-8") as f:

        for tokenlist in parse_incr(f):

            words = []
            pos_tags = []

            for token in tokenlist:

                # ignore multiword tokens
                if isinstance(token["id"], int):

                    words.append(token["form"])
                    pos_tags.append(token["upos"])

            sentences.append(words)
            tags.append(pos_tags)

    return sentences, tags