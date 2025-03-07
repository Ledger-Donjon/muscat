def sbox(x: int) -> int:
    pass

def inv_sbox(x: int) -> int:
    pass

def expand_key(key: bytes) -> list[bytes]:
    pass

class State:
    def state(self) -> bytes:
        pass

    def add_round_key(self, round_key: bytes):
        pass

    def sub_bytes(self):
        pass

    def shift_rows(self):
        pass

    def mix_columns(self):
        pass

    def inv_sub_bytes(self):
        pass

    def inv_mix_columns(self):
        pass

