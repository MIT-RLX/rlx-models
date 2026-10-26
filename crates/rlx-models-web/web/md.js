// A small Markdown renderer for model output: headings, paragraphs, bold,
// italic, inline and fenced code, ordered / unordered lists, block quotes,
// links (http/https only), horizontal rules and pipe tables. Everything is
// HTML-escaped before any markup is added, so model text cannot inject tags.
// Tolerates half-written input (it re-renders on every streamed token).

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function inline(s) {
  // Code spans first, so their contents are left alone.
  const code = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => `\u0000${code.push(c) - 1}\u0000`);
  s = esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^\w_])_([^_\s][^_]*)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(code[i])}</code>`);
}

const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

// Nested lists from `items` ({ indent, ordered, start, text }) from `from` on,
// for items indented at least `indent`: an item 2+ columns deeper than the
// one before it starts a sub-list. Returns [html, index after the last used].
function buildList(items, from, indent) {
  let html = "";
  let i = from;
  while (i < items.length && items[i].indent >= indent) {
    const first = items[i];
    const tag = first.ordered ? "ol" : "ul";
    const start = first.ordered && first.start !== 1 ? ` start="${first.start}"` : "";
    html += `<${tag}${start}>`;
    while (i < items.length && items[i].indent >= indent && items[i].indent < first.indent + 2 &&
      items[i].ordered === first.ordered) {
      const it = items[i++];
      let inner = inline(it.text);
      if (i < items.length && items[i].indent >= it.indent + 2) {
        const [sub, next] = buildList(items, i, it.indent + 2);
        inner += sub;
        i = next;
      }
      html += `<li>${inner}</li>`;
    }
    html += `</${tag}>`;
  }
  return [html, i];
}

const isTableRow = (l) => /^\s*\|.*\|\s*$/.test(l);
const isTableRule = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));

export function renderMarkdown(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let para = [];
  const flush = () => {
    if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`);
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Fenced code (an unclosed fence runs to the end — still streaming).
    const fence = /^\s*```\s*([\w+-]*)/.exec(line);
    if (fence) {
      flush();
      const body = [];
      for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) body.push(lines[i]);
      const lang = fence[1] ? ` data-lang="${esc(fence[1])}"` : "";
      out.push(`<pre${lang}><code>${esc(body.join("\n"))}</code></pre>`);
      continue;
    }
    if (!line.trim()) {
      flush();
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      flush();
      out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
      continue;
    }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      flush();
      out.push("<hr>");
      continue;
    }
    if (isTableRow(line) && i + 1 < lines.length && isTableRule(lines[i + 1])) {
      flush();
      const head = cells(line);
      const rows = [];
      for (i += 2; i < lines.length && isTableRow(lines[i]); i++) rows.push(cells(lines[i]));
      i--;
      out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
        rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody></table>");
      continue;
    }
    if (LIST_ITEM.test(line)) {
      flush();
      // The whole list: items at any depth, their indented continuation
      // lines, and blank lines followed by more of the list.
      const items = [];
      for (; i < lines.length; i++) {
        const m = LIST_ITEM.exec(lines[i]);
        if (m) {
          const ordered = /\d/.test(m[2]);
          items.push({ indent: m[1].length, ordered, start: ordered ? parseInt(m[2], 10) : 1, text: m[3] });
        } else if (/^\s+\S/.test(lines[i])) {
          items[items.length - 1].text += " " + lines[i].trim();
        } else if (!lines[i].trim()) {
          let j = i + 1;
          while (j < lines.length && !lines[j].trim()) j++;
          if (j < lines.length && (LIST_ITEM.test(lines[j]) || /^\s+\S/.test(lines[j]))) i = j - 1;
          else break;
        } else break;
      }
      i--;
      out.push(buildList(items, 0, 0)[0]);
      continue;
    }
    if (/^\s*>/.test(line)) {
      flush();
      const quote = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) quote.push(lines[i].replace(/^\s*>\s?/, ""));
      i--;
      out.push(`<blockquote>${renderMarkdown(quote.join("\n"))}</blockquote>`);
      continue;
    }
    para.push(line);
  }
  flush();
  return out.join("");
}
