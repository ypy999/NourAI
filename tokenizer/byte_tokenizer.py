class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer for valid Unicode strings.

    IDs 0..255 are bytes. PAD/BOS/EOS/UNK occupy 256..259. Special IDs are
    omitted when decoding, matching the original public behavior.
    """

    vocab_size = 260
    pad_id, bos_id, eos_id, unk_id = 256, 257, 258, 259

    def encode(self, text, add_bos=False, add_eos=False):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = list(text.encode("utf-8", errors="strict"))
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids, errors="replace"):
        if errors not in ("replace", "strict", "ignore"):
            raise ValueError("errors must be 'replace', 'strict', or 'ignore'")
        raw = bytearray()
        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if 0 <= token_id <= 255:
                raw.append(token_id)
            elif token_id in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
                continue
            else:
                raise ValueError(f"token ID {token_id} is outside the vocabulary")
        return bytes(raw).decode("utf-8", errors=errors)
