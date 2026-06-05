/**
 * Shared header behaviors: live GitHub star counter + language selector.
 * Loaded by every page that includes the .header-github component.
 */
(function () {
  var REPO = 'rohitg00/ai-engineering-from-scratch';
  var CACHE_KEY = 'gh:stars:' + REPO;
  var CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes

  function format(n) {
    if (n >= 10000) return (n / 1000).toFixed(1).replace(/\\.0$/, '') + 'k';
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\\.0$/, '') + 'k';
    return String(n);
  }

  function paint(n) {
    var els = document.querySelectorAll(
      '.header-github .star-count, #starCount, [data-gh-stars="' + REPO + '"]'
    );
    for (var i = 0; i < els.length; i++) {
      els[i].textContent = format(n);
      els[i].removeAttribute('data-loading');
    }
  }

  function readCache() {
    try {
      var raw = localStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      var parsed = JSON.parse(raw);
      if (Date.now() - parsed.t > CACHE_TTL_MS) return null;
      return parsed.n;
    } catch (e) {
      return null;
    }
  }

  function writeCache(n) {
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify({ n: n, t: Date.now() }));
    } catch (e) {}
  }

  function load() {
    var cached = readCache();
    if (cached != null) {
      paint(cached);
      return;
    }
    fetch('https://api.github.com/repos/' + REPO, {
      headers: { Accept: 'application/vnd.github+json' },
    })
      .then(function (r) {
        if (!r.ok) throw new Error('gh ' + r.status);
        return r.json();
      })
      .then(function (data) {
        var n = data.stargazers_count;
        if (typeof n !== 'number') return;
        writeCache(n);
        paint(n);
      })
      .catch(function () {});
  }

  /* --------------------------------------------------------
   * Language Selector (EN / PT-BR)
   * -------------------------------------------------------- */
  var LANG_STORAGE_KEY = 'aifs:lang';

  var TRANSLATIONS = {
    'en': {
      'Lessons': 'Lessons',
      'Contents': 'Contents',
      'Phases': 'Phases',
      'Glossary': 'Glossary',
      'Catalog': 'Catalog',
      'Search': 'Search',
      'Roadmap': 'Roadmap',
      'Prerequisites': 'Prerequisites',
      'Learning Objectives': 'Learning Objectives',
      'Type': 'Type',
      'Languages': 'Languages',
      'Time': 'Time',
      'Back to catalog': 'Back to catalog',
      'Lesson Catalog': 'Lesson Catalog',
      'AI Glossary': 'AI Glossary',
      'Search lessons...': 'Search lessons...',
      'All Phases': 'All Phases',
      'All Status': 'All Status',
      'Phase': 'Phase',
      'Language': 'Language',
      'Status': 'Status',
      'Search terms...': 'Search terms...',
      // Lesson page panels
      'What This Lesson Ships': 'What This Lesson Ships',
      'Run the Code': 'Run the Code',
      'Test Your Understanding': 'Test Your Understanding',
      'Learning Path': 'Learning Path',
      'Continue Learning': 'Continue Learning',
      // Panel subtitles
      'Prompts, skills, and artifacts you can use right now': 'Prompts, skills, and artifacts you can use right now',
      'Executable files from this lesson': 'Executable files from this lesson',
      'Did you get it?': 'Did you get it?',
      // Loading states
      'Loading lesson...': 'Loading lesson...',
      'Loading outputs...': 'Loading outputs...',
      'Loading code files...': 'Loading code files...',
      'Loading description...': 'Loading description...',
      'Rendering diagram...': 'Rendering diagram...',
      // Buttons
      'View on GitHub': 'View on GitHub',
      'Copy command': 'Copy command',
      'Copied!': 'Copied!',
      'Copy': 'Copy',
      'Install': 'Install',
      // Quiz UI
      'Question': 'Question',
      'of': 'of',
      'Complete all questions to see your score': 'Complete all questions to see your score',
      'Perfect score!': 'Perfect score!',
      'Great work!': 'Great work!',
      'Keep studying!': 'Keep studying!',
      'correct': 'correct',
      // Navigation
      'Previous': 'Previous',
      'Next': 'Next',
      'On this page': 'On this page',
      // Learning path
      'You have completed': 'You have completed',
      'lessons in this phase': 'lessons in this phase',
      'Ready for Phase': 'Ready for Phase',
      // Continue panel
      'You finished this phase!': 'You finished this phase!',
      'Browse all Phase': 'Browse all Phase',
      'lessons': 'lessons',
      'Full course catalog': 'Full course catalog',
      // Quiz sections
      'Pre-Lesson Check': 'Pre-Lesson Check',
      'Mid-Lesson Check': 'Mid-Lesson Check',
      'Post-Lesson Quiz': 'Post-Lesson Quiz',
      // Deep quiz prompt
      'Want a deeper quiz? Run': 'Want a deeper quiz? Run',
      'in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed': 'in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed',
      'Run': 'Run',
      'in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed for a personalized learning path': 'in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed for a personalized learning path',
      // Other page strings
      'Every lesson across all 20 phases. Search, filter, sort.': 'Every lesson across all 20 phases. Search, filter, sort.',
      'No lessons match your filters.': 'No lessons match your filters.',
      'No terms match your search.': 'No terms match your search.',
      'What people say': 'What people say',
      'What it actually means': 'What it actually means',
      'None. This is a starting point.': 'None. This is a starting point.',
      'Final destination. End of the curriculum.': 'Final destination. End of the curriculum.',
      'Read': 'Read',
      'Report / Suggest': 'Report / Suggest',
      'Diagram': 'Diagram',
      'Expanded diagram': 'Expanded diagram',
      'Escape': 'Escape',
      'GitHub stars': 'GitHub stars',
      'Toggle theme': 'Toggle theme',
      'Toggle sidebar': 'Toggle sidebar',
      'Skip to content': 'Skip to content',
      'Home': 'Home',
      'Lesson Catalog - AI Engineering from Scratch': 'Lesson Catalog - AI Engineering from Scratch',
      'AI Glossary - AI Engineering from Scratch': 'AI Glossary - AI Engineering from Scratch',
      'Roadmap - AI Engineering from Scratch': 'Roadmap - AI Engineering from Scratch',
      // Error messages
      'No lesson path specified': 'No lesson path specified',
      'Add a ?path= parameter to the URL.': 'Add a ?path= parameter to the URL.',
      'Render error': 'Render error',
      'Loaded the lesson markdown but failed to render it. Details in the browser console.': 'Loaded the lesson markdown but failed to render it. Details in the browser console.',
      'Lesson not found': 'Lesson not found',
      'Could not fetch the lesson at': 'Could not fetch the lesson at',
      'It may not have been written yet.': 'It may not have been written yet.',
      'Back to Home': 'Back to Home',
      // UI buttons
      'Expand': 'Expand',
      'Close': 'Close',
      'Lab Challenge': 'Lab Challenge',
      'Quiz': 'Quiz',
      'Diagram could not be rendered.': 'Diagram could not be rendered.',
      // Output badges
      'Prompt': 'Prompt',
      'Skill': 'Skill',
      'Output': 'Output',
      'Paste into Claude, Cursor, Codex, OpenClaw, Hermes, or any agent that reads prompts': 'Paste into Claude, Cursor, Codex, OpenClaw, Hermes, or any agent that reads prompts',
      'View lesson on GitHub': 'View lesson on GitHub',
      'Complete': 'Complete',
      'Planned': 'Planned',
      'terms': 'terms',
      'Lesson': 'Lesson'
    },
    'pt-br': {
      'Lessons': 'Aulas',
      'Contents': 'Conteudo',
      'Phases': 'Fases',
      'Glossary': 'Glossário',
      'Catalog': 'Catálogo',
      'Search': 'Buscar',
      'Roadmap': 'Roteiro',
      'Prerequisites': 'Pré-requisitos',
      'Learning Objectives': 'Objetivos de Aprendizado',
      'Type': 'Tipo',
      'Languages': 'Linguagens',
      'Time': 'Tempo',
      'Back to catalog': 'Voltar ao catálogo',
      'Lesson Catalog': 'Catálogo de Aulas',
      'AI Glossary': 'Glossário de IA',
      'Search lessons...': 'Buscar aulas...',
      'All Phases': 'Todas as Fases',
      'All Status': 'Todos os Estados',
      'Phase': 'Fase',
      'Language': 'Idioma',
      'Status': 'Estado',
      'Search terms...': 'Buscar termos...',
      // Lesson page panels
      'What This Lesson Ships': 'O que esta aula entrega',
      'Run the Code': 'Rodar o código',
      'Test Your Understanding': 'Teste seu entendimento',
      'Learning Path': 'Trilha de aprendizado',
      'Continue Learning': 'Continuar aprendendo',
      // Panel subtitles
      'Prompts, skills, and artifacts you can use right now': 'Prompts, skills e artefatos que você pode usar agora',
      'Executable files from this lesson': 'Arquivos executáveis desta aula',
      'Did you get it?': 'Você entendeu?',
      // Loading states
      'Loading lesson...': 'Carregando aula...',
      'Loading outputs...': 'Carregando artefatos...',
      'Loading code files...': 'Carregando arquivos de código...',
      'Loading description...': 'Carregando descrição...',
      'Rendering diagram...': 'Renderizando diagrama...',
      // Buttons
      'View on GitHub': 'Ver no GitHub',
      'Copy command': 'Copiar comando',
      'Copied!': 'Copiado!',
      'Copy': 'Copiar',
      'Install': 'Instalar',
      // Quiz UI
      'Question': 'Pergunta',
      'of': 'de',
      'Complete all questions to see your score': 'Responda todas as perguntas para ver sua pontuacao',
      'Perfect score!': 'Pontuação perfeita!',
      'Great work!': 'Ótimo trabalho!',
      'Keep studying!': 'Continue estudando!',
      'correct': 'correto',
      // Navigation
      'Previous': 'Anterior',
      'Next': 'Próximo',
      'On this page': 'Nesta pagina',
      // Learning path
      'You have completed': 'Você completou',
      'lessons in this phase': 'aulas nesta fase',
      'Ready for Phase': 'Pronto para a Fase',
      // Continue panel
      'You finished this phase!': 'Você finalizou esta fase!',
      'Browse all Phase': 'Ver todas as aulas da Fase',
      'lessons': 'aulas',
      'Full course catalog': 'Catálogo completo do curso',
      // Quiz sections
      'Pre-Lesson Check': 'Verificacao Pre-Aula',
      'Mid-Lesson Check': 'Verificacao intermediária',
      'Post-Lesson Quiz': 'Quiz Pos-Aula',
      // Deep quiz prompt
      'Want a deeper quiz? Run': 'Quer um quiz mais profundo? Execute',
      'in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed': 'no Claude, Cursor, Codex, OpenClaw, Hermes ou qualquer agente com as skills do currículo instaladas',
      'Run': 'Execute',
      'in Claude, Cursor, Codex, OpenClaw, Hermes, or any agent with the curriculum skills installed for a personalized learning path': 'no Claude, Cursor, Codex, OpenClaw, Hermes ou qualquer agente com as skills do currículo instaladas para uma trilha personalizada',
      // Other page strings
      'Every lesson across all 20 phases. Search, filter, sort.': 'Todas as aulas das 20 fases. Busque, filtre, ordene.',
      'No lessons match your filters.': 'Nenhuma aula corresponde aos seus filtros.',
      'No terms match your search.': 'Nenhum termo corresponde a sua busca.',
      'What people say': 'O que as pessoas dizem',
      'What it actually means': 'O que realmente significa',
      'None. This is a starting point.': 'Nenhum. Este e o ponto de partida.',
      'Final destination. End of the curriculum.': 'Destino final. Fim do currículo.',
      'Read': 'Ler',
      'Report / Suggest': 'Reportar / Sugerir',
      'Diagram': 'Diagrama',
      'Expanded diagram': 'Diagrama expandido',
      'Escape': 'Fechar',
      'GitHub stars': 'Estrelas no GitHub',
      'Toggle theme': 'Alternar tema',
      'Toggle sidebar': 'Alternar barra lateral',
      'Skip to content': 'Ir para o conteudo',
      'Home': 'Início',
      'Lesson Catalog - AI Engineering from Scratch': 'Catálogo de Aulas - AI Engineering from Scratch',
      'AI Glossary - AI Engineering from Scratch': 'Glossário de IA - AI Engineering from Scratch',
      'Roadmap - AI Engineering from Scratch': 'Roteiro - AI Engineering from Scratch',
      // Error messages
      'No lesson path specified': 'Nenhum caminho de aula especificado',
      'Add a ?path= parameter to the URL.': 'Adicione um parâmetro ?path= a URL.',
      'Render error': 'Erro de renderização',
      'Loaded the lesson markdown but failed to render it. Details in the browser console.': 'O markdown da aula foi carregado, mas falhou ao renderizar. Detalhes no console do navegador.',
      'Lesson not found': 'Aula não encontrada',
      'Could not fetch the lesson at': 'Não foi possivel buscar a aula em',
      'It may not have been written yet.': 'Pode ser que ainda não tenha sido escrita.',
      'Back to Home': 'Voltar ao Início',
      // UI buttons
      'Expand': 'Expandir',
      'Close': 'Fechar',
      'Lab Challenge': 'Desafio de Laboratorio',
      'Quiz': 'Quiz',
      'Diagram could not be rendered.': 'O diagrama não pôde ser renderizado.',
      // Output badges
      'Prompt': 'Prompt',
      'Skill': 'Skill',
      'Output': 'Saida',
      'Paste into Claude, Cursor, Codex, OpenClaw, Hermes, or any agent that reads prompts': 'Cole no Claude, Cursor, Codex, OpenClaw, Hermes ou qualquer agente que leia prompts',
      'View lesson on GitHub': 'Ver aula no GitHub',
      'Complete': 'Completo',
      'Planned': 'Planejado',
      'terms': 'termos',
      'Lesson': 'Aula'
    }
  };

  function getLang() {
    var params = new URLSearchParams(window.location.search);
    var urlLang = params.get('lang');
    if (urlLang && TRANSLATIONS[urlLang]) return urlLang;
    try {
      var stored = localStorage.getItem(LANG_STORAGE_KEY);
      if (stored && TRANSLATIONS[stored]) return stored;
    } catch (e) {}
    return 'en';
  }

  function setLang(lang) {
    try { localStorage.setItem(LANG_STORAGE_KEY, lang); } catch (e) {}
    var url = new URL(window.location.href);
    url.searchParams.set('lang', lang);
    window.history.replaceState({}, '', url.toString());
  }

  function t(key) {
    var lang = getLang();
    return (TRANSLATIONS[lang] && TRANSLATIONS[lang][key]) || key;
  }

  window.AIFSLang = { getLang: getLang, setLang: setLang, t: t, TRANSLATIONS: TRANSLATIONS };

  function applyTranslations() {
    var lang = getLang();
    // Translate nav links
    document.querySelectorAll('.header-nav a').forEach(function (link) {
      var text = link.textContent.trim();
      var translated = t(text);
      if (translated !== text) {
        var hasSvg = false;
        for (var c = 0; c < link.childNodes.length; c++) {
          if (link.childNodes[c].nodeType === 1 && link.childNodes[c].tagName.toLowerCase() === 'svg') {
            hasSvg = true;
            break;
          }
        }
        if (!hasSvg) link.textContent = translated;
      }
    });

    // Translate elements with data-i18n
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      el.textContent = t(el.getAttribute('data-i18n'));
    });

    // Translate placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
      el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
    });

    // Update lang toggle button
    var langBtn = document.getElementById('langToggle');
    if (langBtn) {
      langBtn.textContent = lang === 'en' ? 'PT' : 'EN';
      langBtn.title = lang === 'en' ? 'Mudar para Portugues' : 'Switch to English';
    }

    document.documentElement.lang = lang === 'pt-br' ? 'pt-BR' : 'en';
  }

  function initLangSelector() {
    var headerInner = document.querySelector('.header-inner');
    if (!headerInner || document.getElementById('langToggle')) return;

    var btn = document.createElement('button');
    btn.id = 'langToggle';
    btn.className = 'lang-toggle';
    btn.type = 'button';
    btn.setAttribute('aria-label', 'Toggle language');
    btn.setAttribute('title', getLang() === 'en' ? 'Mudar para Portugues' : 'Switch to English');

    var themeToggle = document.getElementById('themeToggle');
    if (themeToggle && themeToggle.parentNode === headerInner) {
      headerInner.insertBefore(btn, themeToggle);
    } else {
      headerInner.appendChild(btn);
    }

    btn.addEventListener('click', function () {
      var current = getLang();
      var next = current === 'en' ? 'pt-br' : 'en';
      setLang(next);
      applyTranslations();
      window.location.reload();
    });

    applyTranslations();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      load();
      initLangSelector();
    });
  } else {
    load();
    initLangSelector();
  }
})();
