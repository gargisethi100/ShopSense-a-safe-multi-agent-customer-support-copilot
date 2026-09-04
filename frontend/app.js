/* ShopSense — the shopper's side of the conversation.
 *
 * WHERE THE TRUTH LIVES: not here. The conversation is checkpointed in
 * Postgres behind the API, and this page reads it back. That is why a
 * refresh mid-refund restores exactly what was on screen, and why a
 * second tab shows the same thing. A local copy of the messages would be
 * a second source of truth, and the two would eventually disagree.
 *
 * WHY THERE IS A POLL: POST /chat can tell us a refund froze, but only
 * once, and only to the tab that asked. The decision is then made by
 * someone else, somewhere else. GET /conversations/{id}/state is a cheap
 * read of the checkpointer, so asking it every few seconds while frozen
 * is the whole of our real-time story. No websocket needed for an event
 * that happens once a conversation.
 */

const el = {
  page: document.getElementById('page'),
  opening: document.getElementById('opening'),
  transcript: document.getElementById('transcript'),
  working: document.getElementById('working'),
  hold: document.getElementById('hold'),
  holdAmount: document.getElementById('hold-amount'),
  holdItem: document.getElementById('hold-item'),
  holdOrder: document.getElementById('hold-order'),
  holdRef: document.getElementById('hold-ref'),
  notice: document.getElementById('notice'),
  composer: document.getElementById('composer'),
  entry: document.getElementById('entry'),
  send: document.getElementById('send'),
};

const THREAD_KEY = 'shopsense.thread';
const POLL_MS = 3000;

let threadId = sessionStorage.getItem(THREAD_KEY);
let policies = new Map();   // "RET-1" -> {title, source}
let held = false;
let busy = false;
let pollTimer = null;

// ---------------------------------------------------------------------
// Talking to the API
// ---------------------------------------------------------------------

async function api(path, options) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = new Error(`${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// ---------------------------------------------------------------------
// Rendering
//
// Every string below reaches the DOM through textContent. Answers are
// model output quoting a customer's own words back at them, so treating
// them as markup is how a support chat becomes an XSS hole.
// ---------------------------------------------------------------------

// The agents answer in light Markdown - "**Status:** Delivered", bulleted
// order summaries - because that is what the prompts in agents/ ask for.
// So this renders exactly that much of it: bold, bullets, paragraphs. Not
// a Markdown library, and deliberately not a general one: every node here
// is built by hand, so there is no path from model output to markup.
const INLINE = /\*\*(.+?)\*\*|\[([A-Z]+-\d+)\]/g;
const BULLET = /^\s*[-*]\s+/;

function addCustomerTurn(text) {
  const li = document.createElement('li');
  li.className = 'turn turn-customer';
  li.textContent = text;
  el.transcript.append(li);
}

function appendInline(parent, text, numberFor) {
  let cursor = 0;
  for (const match of text.matchAll(INLINE)) {
    const [, bold, citation] = match;
    let before = text.slice(cursor, match.index);
    // The model writes "within 30 days [RET-1]." A footnote mark belongs
    // against the word it qualifies, so the space in front of it goes.
    if (bold === undefined) before = before.replace(/\s+$/, '');
    parent.append(document.createTextNode(before));
    if (bold !== undefined) {
      const strong = document.createElement('strong');
      strong.textContent = bold;
      parent.append(strong);
    } else if (policies.has(citation)) {
      const sup = document.createElement('sup');
      sup.className = 'cite';
      sup.textContent = String(numberFor(citation));
      parent.append(sup);
    }
    // A citation id we cannot resolve is dropped rather than shown raw:
    // a bracketed code is noise to the person reading it.
    cursor = match.index + match[0].length;
  }
  parent.append(document.createTextNode(text.slice(cursor)));
}

function addAssistantTurn(text) {
  const li = document.createElement('li');
  li.className = 'turn turn-assistant';

  // Citation marks are numbered per reply, in the order they appear, so
  // the same rule cited twice keeps one number.
  const order = [];
  const numberFor = (id) => {
    let i = order.indexOf(id);
    if (i === -1) { order.push(id); i = order.length - 1; }
    return i + 1;
  };

  // Group the lines into blocks: a run of bullets becomes one list,
  // anything else becomes a paragraph.
  const blocks = [];
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    const isBullet = BULLET.test(line);
    const last = blocks[blocks.length - 1];
    if (isBullet && last?.type === 'list') {
      last.items.push(line.replace(BULLET, ''));
    } else if (isBullet) {
      blocks.push({ type: 'list', items: [line.replace(BULLET, '')] });
    } else {
      blocks.push({ type: 'para', text: line });
    }
  }

  blocks.forEach((block, i) => {
    const last = i === blocks.length - 1;
    if (block.type === 'list') {
      const ul = document.createElement('ul');
      ul.className = last ? 'bullets bullets-last' : 'bullets';
      for (const item of block.items) {
        const item_el = document.createElement('li');
        appendInline(item_el, item, numberFor);
        ul.append(item_el);
      }
      li.append(ul);
    } else {
      const p = document.createElement('p');
      p.className = last ? 'para para-last' : 'para';
      appendInline(p, block.text, numberFor);
      li.append(p);
    }
  });

  if (order.length) {
    const notes = document.createElement('ol');
    notes.className = 'notes';
    order.forEach((id, i) => {
      const rule = policies.get(id);
      const note = document.createElement('li');
      note.className = 'note';
      const num = document.createElement('span');
      num.className = 'note-num';
      num.textContent = String(i + 1);
      note.append(num, document.createTextNode(`${rule.title} (${rule.source})`));
      notes.append(note);
    });
    li.append(notes);
  }

  el.transcript.append(li);
}

function renderTranscript(messages) {
  el.transcript.replaceChildren();
  for (const m of messages) {
    if (m.role === 'customer') addCustomerTurn(m.text);
    else addAssistantTurn(m.text);
  }
  if (messages.length) el.opening.hidden = true;
}

function showHold(approval) {
  el.holdAmount.textContent = `$${approval.amount_usd}`;
  el.holdItem.textContent = approval.product_name;
  el.holdOrder.textContent = `Order ${approval.order_id}`;
  el.holdRef.textContent = `Reference ${approval.refund_id}`;
  el.hold.hidden = false;
  setHeld(true);
}

function clearHold() {
  el.hold.hidden = true;
  setHeld(false);
}

function setHeld(value) {
  held = value;
  el.entry.disabled = value;
  updateComposer();
  if (value) {
    el.entry.placeholder = 'Waiting on approval';
    startPolling();
  } else {
    el.entry.placeholder = 'Ask about an order or a policy';
    stopPolling();
  }
}

function updateComposer() {
  el.send.disabled = held || busy || !el.entry.value.trim();
}

function say(text) {
  el.notice.textContent = text;
  el.notice.hidden = !text;
}

function scrollToEnd() {
  el.page.scrollTop = el.page.scrollHeight;
}

// ---------------------------------------------------------------------
// Sending a message
// ---------------------------------------------------------------------

async function send(text) {
  if (busy || held || !text) return true;

  say('');
  el.opening.hidden = true;
  addCustomerTurn(text);

  busy = true;
  updateComposer();
  el.working.hidden = false;
  scrollToEnd();

  try {
    const res = await api('/chat', {
      method: 'POST',
      body: JSON.stringify({ message: text, thread_id: threadId }),
    });

    threadId = res.thread_id;
    sessionStorage.setItem(THREAD_KEY, threadId);

    if (res.pending_approval) {
      showHold(res.pending_approval);
    } else if (res.answer) {
      addAssistantTurn(res.answer);
    } else {
      // A turn with no answer and no approval means the guardrail stopped
      // it without a reply to show. Say so rather than leaving a gap.
      say('That one did not go through. Try asking about an order or one of our policies.');
    }
    return true;
  } catch (err) {
    if (err.status === 422) {
      say('That message was too long to send. Shorten it and try again.');
    } else if (err.status >= 500) {
      // The turn reached the server and broke there, so "check your
      // connection" would send the customer chasing the wrong thing.
      say('Something went wrong at our end on that one. Ask again, or rephrase it — nothing was charged or changed.');
    } else {
      say('That did not reach us. Check your connection and send it again — your message is still in the box below.');
    }

    // A failed turn may still have been recorded before it broke, so do
    // not guess at what happened: re-read the transcript and let the
    // checkpointer say. Whether the customer has to retype follows from
    // that, rather than from our optimism.
    return threadId ? await resync(text) : false;
  } finally {
    busy = false;
    el.working.hidden = true;
    updateComposer();
    scrollToEnd();
  }
}

/** Re-read the conversation. Returns true if `text` is on the record. */
async function resync(text) {
  try {
    const messages = await api(`/conversations/${encodeURIComponent(threadId)}`);
    renderTranscript(messages);
    const lastCustomer = [...messages].reverse().find((m) => m.role === 'customer');
    return lastCustomer?.text === text;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------
// Polling while frozen
// ---------------------------------------------------------------------

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(checkHold, POLL_MS);
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
}

async function checkHold() {
  // A hidden tab is nobody watching. Skip the round trip and pick it up
  // when they come back.
  if (document.hidden || !threadId) return;
  try {
    const state = await api(`/conversations/${encodeURIComponent(threadId)}/state`);
    if (!state.held) {
      clearHold();
      renderTranscript(await api(`/conversations/${encodeURIComponent(threadId)}`));
      scrollToEnd();
    }
  } catch {
    // A failed poll is not worth telling the customer about; the next
    // tick is three seconds away.
  }
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden && held) checkHold();
});

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

async function boot() {
  try {
    const sections = await api('/policies');
    policies = new Map(sections.map((s) => [s.section_id, s]));
  } catch {
    // Without the index, citations are simply dropped from replies. The
    // conversation still works, so this is not worth a message.
  }

  if (!threadId) return;

  try {
    const [messages, state] = await Promise.all([
      api(`/conversations/${encodeURIComponent(threadId)}`),
      api(`/conversations/${encodeURIComponent(threadId)}/state`),
    ]);
    renderTranscript(messages);
    if (state.pending_approval) showHold(state.pending_approval);
    scrollToEnd();
  } catch (err) {
    if (err.status === 404) {
      // The server has never heard of this conversation — most likely a
      // tab left open across a redeploy. Start clean and say so.
      sessionStorage.removeItem(THREAD_KEY);
      threadId = null;
      say('That conversation is no longer on file. This is a fresh start.');
    } else {
      say('Could not load the earlier part of this conversation. You can still ask a new question.');
    }
  }
}

// ---------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------

el.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = el.entry.value.trim();
  el.entry.value = '';
  autoGrow();
  updateComposer();
  send(text).then((ok) => {
    if (ok) return;
    // Failed sends put the words back rather than making the customer
    // retype them. The turn is also dropped from the transcript, because
    // the server has no record of it either.
    el.entry.value = text;
    autoGrow();
    updateComposer();
  });
});

el.entry.addEventListener('input', () => {
  autoGrow();
  updateComposer();
});

el.entry.addEventListener('keydown', (e) => {
  // Enter sends, Shift+Enter starts a new line — what people expect of a
  // message box, and what a bare <textarea> does not do on its own.
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    el.composer.requestSubmit();
  }
});

for (const button of document.querySelectorAll('.opener')) {
  button.addEventListener('click', () => send(button.dataset.say));
}

function autoGrow() {
  el.entry.style.height = 'auto';
  el.entry.style.height = `${el.entry.scrollHeight}px`;
}

updateComposer();
boot();
