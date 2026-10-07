"""Minimal reader for R .rds files (XDR serialization, gzip) containing a data.frame.
Supports REALSXP, INTSXP, LGLSXP, STRSXP, VECSXP, attributes, factors, ALTREP compact sequences (row.names)."""
import gzip, lzma, bz2, struct, numpy as np, pandas as pd, sys

NILVALUE_SXP=254; REFSXP=255; GLOBALENV_SXP=253; EMPTYENV_SXP=242; BASEENV_SXP=241; MISSINGARG_SXP=251; UNBOUNDVALUE_SXP=252
ALTREP_SXP=238; ATTRLISTSXP=239; ATTRLANGSXP=240; BASENAMESPACE_SXP=247; NAMESPACESXP=249; PACKAGESXP=250; PERSISTSXP=248
SYMSXP=1; LISTSXP=2; CLOSXP=3; ENVSXP=4; PROMSXP=5; LANGSXP=6; SPECIALSXP=7; BUILTINSXP=8; CHARSXP=9; LGLSXP=10; INTSXP=13; REALSXP=14; CPLXSXP=15; STRSXP=16; DOTSXP=17; VECSXP=19; EXPRSXP=20; RAWSXP=24; S4SXP=25
NA_INT = -2147483648

class LazyStr:
    def __init__(self, b, start, n): self.b=b; self.start=start; self.n=n
    def decode(self):
        res = np.empty(self.n, dtype=object); b=self.b; p=self.start; unpack=struct.unpack_from
        for i in range(self.n):
            ln = unpack('>i', b, p+4)[0]; p += 8
            if ln == -1: res[i] = None
            else:
                sb = b[p:p+ln]; p += ln
                try: res[i] = sb.decode('utf-8')
                except UnicodeDecodeError: res[i] = sb.decode('latin-1')
        return res
    def __len__(self): return self.n

class R:
    def __init__(self, data, keep=None):
        self.b = data; self.p = 0; self.refs = []; self.keep = keep
    def i4(self):
        v = struct.unpack_from('>i', self.b, self.p)[0]; self.p += 4; return v
    def ints(self, n):
        a = np.frombuffer(self.b, dtype='>i4', count=n, offset=self.p); self.p += 4*n; return a.astype(np.int32)
    def dbls(self, n):
        a = np.frombuffer(self.b, dtype='>f8', count=n, offset=self.p); self.p += 8*n; return a.astype(np.float64)
    def length(self):
        n = self.i4()
        if n == -1:
            hi = self.i4(); lo = self.i4(); n = (hi << 32) + lo
        return n
    def charsxp(self, flags):
        n = self.i4()
        if n == -1: return None
        s = self.b[self.p:self.p+n]; self.p += n
        try: return s.decode('utf-8')
        except UnicodeDecodeError: return s.decode('latin-1')
    def item(self):
        flags = self.i4()
        t = flags & 0xFF; has_attr = (flags >> 9) & 1; has_tag = (flags >> 10) & 1; is_obj = (flags >> 8) & 1
        if t == NILVALUE_SXP: return None
        if t == REFSXP:
            idx = flags >> 8
            if idx == 0: idx = self.i4()
            return self.refs[idx-1]
        if t in (GLOBALENV_SXP, EMPTYENV_SXP, BASEENV_SXP, MISSINGARG_SXP, UNBOUNDVALUE_SXP, BASENAMESPACE_SXP): return None
        if t == SYMSXP:
            s = self.item(); self.refs.append(s); return s
        if t in (LISTSXP, LANGSXP, PROMSXP, DOTSXP, ATTRLISTSXP, ATTRLANGSXP):
            attr = self.item() if has_attr else None
            tag = self.item() if has_tag else None
            car = self.item(); cdr = self.item()
            out = [(tag, car)]
            if isinstance(cdr, list): out += cdr
            return out
        if t == CHARSXP: return self.charsxp(flags)
        if t == ALTREP_SXP:
            info = self.item(); state = self.item(); attr = self.item()
            cls = info[0][1] if isinstance(info, list) else None
            name = cls if isinstance(cls, str) else (cls[0] if isinstance(cls, list) else None)
            if name in ('compact_intseq',):
                n, start, step = state; return np.arange(int(start), int(start)+int(n)*int(step), int(step), dtype=np.int32)
            if name in ('compact_realseq',):
                n, start, step = state; return np.arange(start, start+n*step, step)
            if name in ('wrap_integer','wrap_real','wrap_logical','wrap_string','wrap_list'):
                return state[0][1] if isinstance(state, list) else state
            if name == 'deferred_string':
                v = state[0][1] if isinstance(state, list) else state
                return np.array([str(x) for x in v], dtype=object)
            return state
        if t == LGLSXP:
            n = self.length(); v = self.ints(n); out = v.astype(float); out[v == NA_INT] = np.nan; res = out
        elif t == INTSXP:
            n = self.length(); v = self.ints(n); res = v
        elif t == REALSXP:
            n = self.length(); res = self.dbls(n)
        elif t == CPLXSXP:
            n = self.length(); res = self.dbls(2*n)
        elif t == STRSXP:
            n = self.length(); b = self.b; p = self.p; start = p
            unpack = struct.unpack_from
            for i in range(n):
                ln = unpack('>i', b, p+4)[0]; p += 8
                if ln > 0: p += ln
            self.p = p
            res = LazyStr(b, start, n)   # decoded only if the column is kept
        elif t in (VECSXP, EXPRSXP):
            n = self.length(); res = [self.item() for _ in range(n)]
        elif t == RAWSXP:
            n = self.length(); res = self.b[self.p:self.p+n]; self.p += n
        elif t == S4SXP:
            res = None
        else:
            raise ValueError(f'unsupported SEXP type {t} at {self.p}')
        attrs = self.item() if has_attr else None
        if attrs:
            d = {k: v for k, v in attrs}
            if isinstance(res, LazyStr) and not isinstance(d.get('class'), (str, list)) and 'names' in d: res = res.decode()
            if 'levels' in d and isinstance(res, np.ndarray):
                lv = d['levels']; lv = np.array(lv.decode() if isinstance(lv, LazyStr) else lv, dtype=object); out = np.empty(len(res), dtype=object)
                m = res != NA_INT; out[m] = lv[res[m]-1]; out[~m] = None; res = out
            if 'names' in d and isinstance(res, list):
                nm = d['names']; nm = nm.decode() if isinstance(nm, LazyStr) else nm
                res = dict(zip(nm, res))
            if 'class' in d and isinstance(res, dict):
                cls = d['class']; cls = cls.decode() if isinstance(cls, LazyStr) else cls; cls = list(cls) if not isinstance(cls, str) else [cls]
                if 'data.frame' in cls:
                    cols = {}
                    for k, v in res.items():
                        if self.keep is not None and k not in self.keep: continue
                        if isinstance(v, LazyStr): v = v.decode()
                        if isinstance(v, np.ndarray) and v.dtype == np.int32:
                            vv = v.astype(float); vv[v == NA_INT] = np.nan; v = vv
                        cols[k] = v
                    res = pd.DataFrame(cols)
        return res

def read_rds(path, keep=None):
    head = open(path,'rb').read(6)
    if head[:2] == b'\x1f\x8b': raw = gzip.open(path, 'rb').read()
    elif head[:6] == b'\xfd7zXZ\x00': raw = lzma.open(path, 'rb').read()
    elif head[:3] == b'BZh': raw = bz2.open(path, 'rb').read()
    else: raw = open(path,'rb').read()
    r = R(raw, set(keep) if keep else None)
    hdr = raw[:2]
    assert hdr == b'X\n', 'only XDR format supported'
    r.p = 2
    ver = r.i4(); wv = r.i4(); minv = r.i4()
    if ver == 3:
        n = r.i4(); r.p += n  # native encoding string
    out = r.item()
    return out.decode() if isinstance(out, LazyStr) else out

if __name__ == '__main__':
    df = read_rds(sys.argv[1]); print(type(df), getattr(df, 'shape', None))
    if isinstance(df, pd.DataFrame): print(df.columns.tolist()[:80]); print(df.head(3).T.head(60))
