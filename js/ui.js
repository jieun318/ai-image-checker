/**
 * ui.js - 판별 기록 관리, 알림 및 공통 UI 인터랙션 헬퍼
 */

class HistoryManager {
  constructor(storageKey = 'ai_detector_history_session') {
    this.storageKey = storageKey;
    this.history = [];
    this.load();
  }

  load() {
    try {
      const saved = sessionStorage.getItem(this.storageKey);
      if (saved) {
        this.history = JSON.parse(saved);
      }
    } catch (e) {
      console.warn('기록 로드 실패:', e);
      this.history = [];
    }
  }

  save() {
    try {
      sessionStorage.setItem(this.storageKey, JSON.stringify(this.history));
    } catch (e) {
      console.warn('기록 저장 실패 (용량 초과 가능성):', e);
      // DataURL이 너무 길어 로컬스토리지가 차면 오래된 항목 삭제 후 재시도
      if (this.history.length > 5) {
        this.history = this.history.slice(0, 5);
        try { sessionStorage.setItem(this.storageKey, JSON.stringify(this.history)); } catch (_) {}
      }
    }
  }

  add({ thumbDataUrl, name, verdict, realProb, aiProb, confidence }) {
    const item = {
      id: Date.now(),
      thumbDataUrl,
      name: name || '업로드 이미지',
      verdict,
      realProb,
      aiProb,
      confidence,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    this.history.unshift(item); // 최신순
    // 세션 기록 최대 20개 유지
    if (this.history.length > 20) {
      this.history.pop();
    }
    this.save();
    return item;
  }

  clear() {
    this.history = [];
    sessionStorage.removeItem(this.storageKey);
  }

  getAll() {
    return this.history;
  }
}

// 썸네일 축소 생성기 (로컬스토리지 용량 절약)
function createThumbnail(imgElement, maxSize = 160) {
  const canvas = document.createElement('canvas');
  let width = imgElement.naturalWidth || imgElement.width || 200;
  let height = imgElement.naturalHeight || imgElement.height || 200;

  if (width > height) {
    if (width > maxSize) {
      height = Math.round((height * maxSize) / width);
      width = maxSize;
    }
  } else {
    if (height > maxSize) {
      width = Math.round((width * maxSize) / height);
      height = maxSize;
    }
  }

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imgElement, 0, 0, width, height);
  return canvas.toDataURL('image/jpeg', 0.7);
}

// 전역 인스턴스
window.historyManager = new HistoryManager();
window.createThumbnail = createThumbnail;
