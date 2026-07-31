export function canUseSpeech() {
  return "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
}

export function speak(text: string, volume: number) {
  if (!canUseSpeech() || !text.trim()) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.volume = volume;
  window.speechSynthesis.speak(utterance);
}

export function stopSpeech() {
  if (canUseSpeech()) window.speechSynthesis.cancel();
}
