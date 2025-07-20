/**
 * @fileoverview Tree visualization utility for XGBoost decision trees
 * 
 * The TreeVisualizer class provides methods to extract, parse, and visualize
 * decision trees from trained XGBoost models in multiple formats.
 */

import {
  TreeNode,
  TreeVisualizationOptions,
  VisualizationFormat,
  TreeVisualizationResult,
  Logger,
} from '../types';

/**
 * Tree visualization utility class
 * 
 * Features:
 * - Extract tree structure from XGBoost models
 * - Multiple visualization formats (text, JSON, HTML, SVG)
 * - Feature name mapping
 * - Configurable depth and precision
 * - Tree statistics and metadata
 * 
 * @example
 * ```typescript
 * const visualizer = new TreeVisualizer(['feature1', 'feature2', 'feature3']);
 * 
 * // Text visualization
 * const textViz = visualizer.visualizeTree(model, { format: 'text', treeIndex: 0 });
 * console.log(textViz.content);
 * 
 * // HTML visualization
 * const htmlViz = visualizer.visualizeTree(model, { format: 'html', includeFeatureNames: true });
 * 
 * // Save to file
 * await visualizer.saveVisualization('./tree.html', model, { format: 'html' });
 * ```
 */
export class TreeVisualizer {
  private featureNames: string[];
  private logger: Logger | undefined;

  /**
   * Creates a new TreeVisualizer instance
   * 
   * @param featureNames - Array of feature names in order
   * @param logger - Optional logger for debugging
   */
  constructor(featureNames: string[], logger?: Logger) {
    this.featureNames = featureNames;
    this.logger = logger;
  }

  /**
   * Visualizes a decision tree from an XGBoost model
   * 
   * @param model - Trained XGBoost model
   * @param options - Visualization options
   * @returns Tree visualization result
   */
  public visualizeTree(
    model: any,
    format: VisualizationFormat,
    options: TreeVisualizationOptions = {}
  ): TreeVisualizationResult {
    this.logger?.info('Generating tree visualization', { format, options });

    // Extract tree data from model
    const modelJSON = model.toJSON();
    const treeIndex = options.treeIndex ?? 0;

    if (!modelJSON.trees || !modelJSON.trees[treeIndex]) {
      throw new Error(`Tree ${treeIndex} not found in model`);
    }

    // Parse tree structure
    const treeData = modelJSON.trees[treeIndex];
    const rootNode = this.parseTreeNode(treeData.root, 0);

    // Calculate tree metadata
    const metadata = this.calculateTreeMetadata(rootNode, treeIndex);

    // Generate visualization based on format
    let content: string;
    switch (format) {
      case 'text':
        content = this.generateTextVisualization(rootNode, options);
        break;
      case 'json':
        content = this.generateJSONVisualization(rootNode, options);
        break;
      case 'html':
        content = this.generateHTMLVisualization(rootNode, options, metadata);
        break;
      case 'svg':
        content = this.generateSVGVisualization(rootNode, options, metadata);
        break;
      case 'uml':
        content = this.generateUMLVisualization(rootNode, options, metadata);
        break;
      default:
        throw new Error(`Unsupported visualization format: ${format}`);
    }

    return {
      format,
      content,
      metadata,
    };
  }

  /**
   * Saves tree visualization to a file
   * 
   * @param filePath - Path to save the visualization
   * @param model - Trained XGBoost model
   * @param format - Visualization format
   * @param options - Visualization options
   */
  public async saveVisualization(
    filePath: string,
    model: any,
    format: VisualizationFormat,
    options: TreeVisualizationOptions = {}
  ): Promise<void> {
    const result = this.visualizeTree(model, format, options);
    
    const fs = require('fs');
    fs.writeFileSync(filePath, result.content, 'utf8');
    
    this.logger?.info('Tree visualization saved', { filePath, format });
  }

  /**
   * Gets tree statistics from a trained model
   * 
   * @param model - Trained XGBoost model
   * @returns Tree statistics
   */
  public getTreeStatistics(model: any): {
    treeCount: number;
    trees: Array<{
      index: number;
      maxDepth: number;
      nodeCount: number;
      leafCount: number;
    }>;
  } {
    const modelJSON = model.toJSON();
    
    if (!modelJSON.trees) {
      return { treeCount: 0, trees: [] };
    }

    const trees = modelJSON.trees.map((treeData: any, index: number) => {
      const rootNode = this.parseTreeNode(treeData.root, 0);
      const metadata = this.calculateTreeMetadata(rootNode, index);
      
      return {
        index,
        maxDepth: metadata.maxDepth,
        nodeCount: metadata.nodeCount,
        leafCount: metadata.leafCount,
      };
    });

    return {
      treeCount: modelJSON.trees.length,
      trees,
    };
  }

  /**
   * Parses a tree node from XGBoost JSON format
   * 
   * @private
   */
  private parseTreeNode(nodeData: any, depth: number, nodeId: string = '0'): TreeNode {
    if (!nodeData) {
      throw new Error('Invalid node data');
    }

    const node: TreeNode = {
      featureIndex: nodeData.featureIndex,
      threshold: nodeData.threshold,
      left: null,
      right: null,
      value: nodeData.value,
      isLeaf: nodeData.isLeaf || nodeData.featureIndex === null,
      depth,
      nodeId,
    };

    // Parse child nodes recursively
    if (nodeData.left) {
      node.left = this.parseTreeNode(nodeData.left, depth + 1, `${nodeId}.0`);
    }

    if (nodeData.right) {
      node.right = this.parseTreeNode(nodeData.right, depth + 1, `${nodeId}.1`);
    }

    return node;
  }

  /**
   * Calculates tree metadata
   * 
   * @private
   */
  private calculateTreeMetadata(rootNode: TreeNode, treeIndex: number): {
    treeIndex: number;
    maxDepth: number;
    nodeCount: number;
    leafCount: number;
  } {
    let maxDepth = 0;
    let nodeCount = 0;
    let leafCount = 0;

    const traverse = (node: TreeNode) => {
      nodeCount++;
      maxDepth = Math.max(maxDepth, node.depth || 0);

      if (node.isLeaf) {
        leafCount++;
      }

      if (node.left) traverse(node.left);
      if (node.right) traverse(node.right);
    };

    traverse(rootNode);

    return {
      treeIndex,
      maxDepth,
      nodeCount,
      leafCount,
    };
  }

  /**
   * Generates text-based tree visualization
   * 
   * @private
   */
  private generateTextVisualization(rootNode: TreeNode, options: TreeVisualizationOptions): string {
    const lines: string[] = [];
    const precision = options.precision ?? 4;
    const includeValues = options.includeValues ?? true;
    const maxDepth = options.maxDepth ?? Infinity;

    const traverse = (node: TreeNode, prefix: string = '', isLast: boolean = true) => {
      if ((node.depth || 0) > maxDepth) return;

      const connector = isLast ? '└── ' : '├── ';
      let label = '';

      if (node.isLeaf) {
        label = includeValues ? `Leaf: ${node.value?.toFixed(precision)}` : 'Leaf';
      } else {
        const featureName = options.includeFeatureNames && node.featureIndex !== null
          ? this.featureNames[node.featureIndex] || `Feature_${node.featureIndex}`
          : `Feature_${node.featureIndex}`;
        
        label = `${featureName} <= ${node.threshold?.toFixed(precision)}`;
      }

      lines.push(prefix + connector + label);

      if (!node.isLeaf && (node.depth || 0) < maxDepth) {
        const newPrefix = prefix + (isLast ? '    ' : '│   ');
        
        if (node.left) {
          traverse(node.left, newPrefix, !node.right);
        }
        if (node.right) {
          traverse(node.right, newPrefix, true);
        }
      }
    };

    lines.push('Decision Tree');
    traverse(rootNode);
    
    return lines.join('\n');
  }

  /**
   * Generates JSON tree visualization
   * 
   * @private
   */
  private generateJSONVisualization(rootNode: TreeNode, options: TreeVisualizationOptions): string {
    const precision = options.precision ?? 4;
    const maxDepth = options.maxDepth ?? Infinity;

    const convertNode = (node: TreeNode): any => {
      if ((node.depth || 0) > maxDepth) return null;

      const result: any = {
        nodeId: node.nodeId,
        depth: node.depth,
        isLeaf: node.isLeaf,
      };

      if (node.isLeaf) {
        if (options.includeValues !== false) {
          result.value = node.value !== null ? Number(node.value.toFixed(precision)) : null;
        }
      } else {
        result.featureIndex = node.featureIndex;
        if (options.includeFeatureNames && node.featureIndex !== null) {
          result.featureName = this.featureNames[node.featureIndex] || `Feature_${node.featureIndex}`;
        }
        result.threshold = node.threshold !== null ? Number(node.threshold.toFixed(precision)) : null;
        
        if (node.left) result.left = convertNode(node.left);
        if (node.right) result.right = convertNode(node.right);
      }

      return result;
    };

    return JSON.stringify(convertNode(rootNode), null, 2);
  }

  /**
   * Generates HTML tree visualization
   * 
   * @private
   */
  private generateHTMLVisualization(
    rootNode: TreeNode,
    options: TreeVisualizationOptions,
    metadata: any
  ): string {
    const precision = options.precision ?? 4;
    const maxDepth = options.maxDepth ?? Infinity;

    const generateNodeHTML = (node: TreeNode): string => {
      if ((node.depth || 0) > maxDepth) return '';

      const indent = '  '.repeat((node.depth || 0) + 2);
      
      if (node.isLeaf) {
        const value = options.includeValues !== false ? 
          `: ${node.value?.toFixed(precision)}` : '';
        return `${indent}<div class="leaf-node">🍃 Leaf${value}</div>\n`;
      }

      const featureName = options.includeFeatureNames && node.featureIndex !== null
        ? this.featureNames[node.featureIndex] || `Feature_${node.featureIndex}`
        : `Feature_${node.featureIndex}`;

      const threshold = node.threshold?.toFixed(precision);
      
      let html = `${indent}<div class="split-node">\n`;
      html += `${indent}  <div class="split-condition">📊 ${featureName} ≤ ${threshold}</div>\n`;
      
      if (node.left || node.right) {
        html += `${indent}  <div class="children">\n`;
        if (node.left) {
          html += `${indent}    <div class="left-child">\n`;
          html += `${indent}      <div class="branch-label">← True</div>\n`;
          html += generateNodeHTML(node.left);
          html += `${indent}    </div>\n`;
        }
        if (node.right) {
          html += `${indent}    <div class="right-child">\n`;
          html += `${indent}      <div class="branch-label">→ False</div>\n`;
          html += generateNodeHTML(node.right);
          html += `${indent}    </div>\n`;
        }
        html += `${indent}  </div>\n`;
      }
      
      html += `${indent}</div>\n`;
      return html;
    };

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Decision Tree Visualization</title>
  <style>
    body {
      font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
      margin: 20px;
      background-color: #f5f5f5;
    }
    .tree-container {
      background: white;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .tree-header {
      margin-bottom: 20px;
      padding: 15px;
      background: #2563eb;
      color: white;
      border-radius: 6px;
    }
    .tree-stats {
      display: flex;
      gap: 20px;
      margin-top: 10px;
      font-size: 14px;
    }
    .split-node {
      margin: 8px 0;
      border-left: 3px solid #3b82f6;
      padding-left: 12px;
    }
    .split-condition {
      font-weight: bold;
      color: #1e40af;
      margin-bottom: 5px;
    }
    .leaf-node {
      margin: 8px 0;
      padding: 8px 12px;
      background: #dcfce7;
      border: 1px solid #16a34a;
      border-radius: 4px;
      color: #166534;
      font-weight: 500;
    }
    .children {
      margin-left: 15px;
    }
    .left-child, .right-child {
      margin: 10px 0;
    }
    .branch-label {
      font-size: 12px;
      color: #6b7280;
      font-weight: 600;
      margin-bottom: 5px;
    }
  </style>
</head>
<body>
  <div class="tree-container">
    <div class="tree-header">
      <h1>🌳 Decision Tree Visualization</h1>
      <div class="tree-stats">
        <span>Tree Index: ${metadata.treeIndex}</span>
        <span>Max Depth: ${metadata.maxDepth}</span>
        <span>Nodes: ${metadata.nodeCount}</span>
        <span>Leaves: ${metadata.leafCount}</span>
      </div>
    </div>
    <div class="tree-content">
${generateNodeHTML(rootNode)}
    </div>
  </div>
</body>
</html>`;
  }

  /**
   * Generates SVG tree visualization
   * 
   * @private
   */
  private generateSVGVisualization(
    rootNode: TreeNode,
    options: TreeVisualizationOptions,
    metadata: any
  ): string {
    const precision = options.precision ?? 4;
    const maxDepth = options.maxDepth ?? Infinity;
    
    // Calculate SVG dimensions
    const nodeWidth = 180;
    const nodeHeight = 60;
    const levelHeight = 100;
    const maxWidth = Math.pow(2, Math.min(metadata.maxDepth, maxDepth)) * nodeWidth;
    const maxHeight = (Math.min(metadata.maxDepth, maxDepth) + 1) * levelHeight + 100;

    let svgContent = '';

    const generateNodeSVG = (node: TreeNode, x: number, y: number, width: number): void => {
      if ((node.depth || 0) > maxDepth) return;

      const nodeX = x - nodeWidth / 2;
      const nodeY = y - nodeHeight / 2;

      if (node.isLeaf) {
        // Leaf node
        const value = options.includeValues !== false ? 
          `${node.value?.toFixed(precision)}` : 'Leaf';
        
        svgContent += `
        <rect x="${nodeX}" y="${nodeY}" width="${nodeWidth}" height="${nodeHeight}" 
              fill="#dcfce7" stroke="#16a34a" stroke-width="2" rx="8"/>
        <text x="${x}" y="${y - 5}" text-anchor="middle" font-size="12" font-weight="bold" fill="#166534">
          🍃 Leaf
        </text>
        <text x="${x}" y="${y + 10}" text-anchor="middle" font-size="11" fill="#166534">
          ${value}
        </text>`;
      } else {
        // Split node
        const featureName = options.includeFeatureNames && node.featureIndex !== null
          ? this.featureNames[node.featureIndex] || `F${node.featureIndex}`
          : `F${node.featureIndex}`;
        
        const threshold = node.threshold?.toFixed(precision);
        
        svgContent += `
        <rect x="${nodeX}" y="${nodeY}" width="${nodeWidth}" height="${nodeHeight}" 
              fill="#dbeafe" stroke="#3b82f6" stroke-width="2" rx="8"/>
        <text x="${x}" y="${y - 8}" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e40af">
          📊 ${featureName}
        </text>
        <text x="${x}" y="${y + 8}" text-anchor="middle" font-size="10" fill="#1e40af">
          ≤ ${threshold}
        </text>`;

        // Draw connections and child nodes
        if (node.left || node.right) {
          const childY = y + levelHeight;
          const childWidth = width / 2;

          if (node.left) {
            const leftX = x - width / 4;
            // Connection line
            svgContent += `
            <line x1="${x}" y1="${y + nodeHeight/2}" x2="${leftX}" y2="${childY - nodeHeight/2}" 
                  stroke="#059669" stroke-width="2"/>
            <text x="${(x + leftX) / 2 - 15}" y="${(y + childY) / 2}" font-size="10" fill="#059669" font-weight="bold">
              True
            </text>`;
            generateNodeSVG(node.left, leftX, childY, childWidth);
          }

          if (node.right) {
            const rightX = x + width / 4;
            // Connection line
            svgContent += `
            <line x1="${x}" y1="${y + nodeHeight/2}" x2="${rightX}" y2="${childY - nodeHeight/2}" 
                  stroke="#dc2626" stroke-width="2"/>
            <text x="${(x + rightX) / 2 + 15}" y="${(y + childY) / 2}" font-size="10" fill="#dc2626" font-weight="bold">
              False
            </text>`;
            generateNodeSVG(node.right, rightX, childY, childWidth);
          }
        }
      }
    };

    generateNodeSVG(rootNode, maxWidth / 2, nodeHeight, maxWidth);

    return `<svg width="${maxWidth}" height="${maxHeight}" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'Monaco', 'Menlo', 'Consolas', monospace; }
  </style>
  
  <!-- Title -->
  <rect x="10" y="10" width="${maxWidth - 20}" height="40" fill="#2563eb" rx="6"/>
  <text x="${maxWidth / 2}" y="35" text-anchor="middle" font-size="16" font-weight="bold" fill="white">
    🌳 Decision Tree (Index: ${metadata.treeIndex}, Depth: ${metadata.maxDepth}, Nodes: ${metadata.nodeCount})
  </text>
  
  <!-- Tree content -->
  <g transform="translate(0, 60)">
${svgContent}
  </g>
</svg>`;
  }

  /**
   * Generates UML tree visualization in flowchart style
   * 
   * @private
   */
  private generateUMLVisualization(
    rootNode: TreeNode,
    options: TreeVisualizationOptions,
    metadata: any
  ): string {
    const precision = options.precision ?? 4;
    const maxDepth = options.maxDepth ?? Infinity;

    // First, collect all nodes and their relationships
    const nodes: Array<{node: TreeNode, x: number, y: number}> = [];
    const connections: Array<{from: string, to: string, label: string, type: 'true' | 'false'}> = [];
    
    // Calculate node positions for flowchart layout
    const levelWidth = 300;
    const levelHeight = 150;
    const nodeWidth = 250;
    const nodeHeight = 120;
    
    const collectNodes = (node: TreeNode, depth: number, position: number) => {
      if (depth > maxDepth) return;
      
      const x = depth * levelWidth + 50;
      const y = position * levelHeight + 100;
      
      nodes.push({ node, x, y });
      
      let childPosition = position * 2;
      
      if (node.left) {
        connections.push({
          from: node.nodeId || '0',
          to: node.left.nodeId || '0.0',
          label: `≤ ${node.threshold?.toFixed(precision)}`,
          type: 'true'
        });
        collectNodes(node.left, depth + 1, childPosition);
      }
      
      if (node.right) {
        connections.push({
          from: node.nodeId || '0',
          to: node.right.nodeId || '0.1',
          label: `> ${node.threshold?.toFixed(precision)}`,
          type: 'false'
        });
        collectNodes(node.right, depth + 1, childPosition + 1);
      }
    };
    
    collectNodes(rootNode, 0, 0);
    
    // Generate flowchart nodes
    const generateFlowchartNodes = () => {
      return nodes.map(({node, x, y}) => {
        const nodeId = node.nodeId || '0';
        
        if (node.isLeaf) {
          const value = options.includeValues !== false ? 
            `${node.value?.toFixed(precision)}` : 'Result';
          
          return `
          <g class="flowchart-node leaf-node" id="node-${nodeId}" transform="translate(${x}, ${y})">
            <rect class="node-bg leaf-bg" width="${nodeWidth}" height="${nodeHeight}" rx="15"/>
            <circle class="node-icon" cx="30" cy="30" r="15" fill="#27ae60"/>
            <text class="node-icon-text" x="30" y="35">🍃</text>
            <text class="node-title" x="60" y="25">Leaf Node</text>
            <text class="node-id" x="60" y="45">${nodeId}</text>
            <line class="separator" x1="20" y1="55" x2="${nodeWidth-20}" y2="55"/>
            <text class="property-label" x="30" y="75">Value:</text>
            <text class="property-value" x="30" y="95">${value}</text>
            <text class="property-label" x="150" y="75">Depth:</text>
            <text class="property-value" x="150" y="95">${node.depth}</text>
          </g>`;
        } else {
          const featureName = options.includeFeatureNames && node.featureIndex !== null
            ? this.featureNames[node.featureIndex] || `Feature_${node.featureIndex}`
            : `Feature_${node.featureIndex}`;
          
          const threshold = node.threshold?.toFixed(precision);
          
          return `
          <g class="flowchart-node decision-node" id="node-${nodeId}" transform="translate(${x}, ${y})">
            <rect class="node-bg decision-bg" width="${nodeWidth}" height="${nodeHeight}" rx="15"/>
            <circle class="node-icon" cx="30" cy="30" r="15" fill="#3498db"/>
            <text class="node-icon-text" x="30" y="35">📊</text>
            <text class="node-title" x="60" y="25">Decision</text>
            <text class="node-id" x="60" y="45">${nodeId}</text>
            <line class="separator" x1="20" y1="55" x2="${nodeWidth-20}" y2="55"/>
            <text class="property-label" x="30" y="75">Feature:</text>
            <text class="property-value" x="30" y="95">${featureName}</text>
            <text class="property-label" x="30" y="110">Threshold: ≤ ${threshold}</text>
          </g>`;
        }
      }).join('');
    };
    
    // Generate flowchart connections/arrows
    const generateFlowchartConnections = () => {
      return connections.map(({from, to, label, type}) => {
        const fromNode = nodes.find(n => n.node.nodeId === from);
        const toNode = nodes.find(n => n.node.nodeId === to);
        
        if (!fromNode || !toNode) return '';
        
        const fromX = fromNode.x + nodeWidth;
        const fromY = fromNode.y + nodeHeight / 2;
        const toX = toNode.x;
        const toY = toNode.y + nodeHeight / 2;
        
        const midX = (fromX + toX) / 2;
        const strokeColor = type === 'true' ? '#27ae60' : '#e74c3c';
        const labelBg = type === 'true' ? '#d5f4e6' : '#fadbd8';
        
        return `
        <g class="flowchart-connection ${type}-connection">
          <defs>
            <marker id="arrowhead-${type}" markerWidth="10" markerHeight="7" 
                    refX="9" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="${strokeColor}"/>
            </marker>
          </defs>
          <path d="M ${fromX} ${fromY} Q ${midX} ${fromY} ${toX} ${toY}" 
                stroke="${strokeColor}" stroke-width="3" fill="none" 
                marker-end="url(#arrowhead-${type})"/>
          <rect x="${midX - 40}" y="${fromY - 15}" width="80" height="25" 
                fill="${labelBg}" stroke="${strokeColor}" stroke-width="1" rx="12"/>
          <text x="${midX}" y="${fromY - 2}" text-anchor="middle" 
                class="connection-label ${type}-label">${label}</text>
        </g>`;
      }).join('');
    };
    
    // Calculate SVG dimensions
    const maxX = Math.max(...nodes.map(n => n.x)) + nodeWidth + 100;
    const maxY = Math.max(...nodes.map(n => n.y)) + nodeHeight + 100;

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UML Flowchart Decision Tree</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      margin: 0;
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    
    .flowchart-container {
      background: white;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
      overflow-x: auto;
    }
    
    .flowchart-header {
      text-align: center;
      margin-bottom: 20px;
      padding: 20px;
      background: linear-gradient(135deg, #2c3e50, #3498db);
      color: white;
      border-radius: 8px;
    }
    
    .flowchart-title {
      font-size: 28px;
      font-weight: bold;
      margin: 0;
    }
    
    .flowchart-stats {
      display: flex;
      justify-content: center;
      gap: 30px;
      margin-top: 15px;
      font-size: 14px;
    }
    
    .flowchart-controls {
      text-align: center;
      margin: 20px 0;
    }
    
    .flowchart-button {
      background: linear-gradient(135deg, #3498db, #2980b9);
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 6px;
      cursor: pointer;
      margin: 0 5px;
      font-weight: bold;
      transition: all 0.3s ease;
    }
    
    .flowchart-button:hover {
      background: linear-gradient(135deg, #2980b9, #1f618d);
      transform: translateY(-1px);
    }
    
    .flowchart-svg {
      border: 2px solid #bdc3c7;
      border-radius: 8px;
      background: #ecf0f1;
    }
    
    .node-bg {
      fill: white;
      stroke: #34495e;
      stroke-width: 2;
      filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2));
    }
    
    .decision-bg {
      fill: #ebf3fd;
      stroke: #3498db;
    }
    
    .leaf-bg {
      fill: #e8f5e8;
      stroke: #27ae60;
    }
    
    .node-icon {
      stroke: white;
      stroke-width: 2;
    }
    
    .node-icon-text {
      font-size: 12px;
      text-anchor: middle;
      dominant-baseline: middle;
      fill: white;
    }
    
    .node-title {
      font-size: 14px;
      font-weight: bold;
      fill: #2c3e50;
      dominant-baseline: middle;
    }
    
    .node-id {
      font-size: 11px;
      fill: #7f8c8d;
      dominant-baseline: middle;
    }
    
    .separator {
      stroke: #bdc3c7;
      stroke-width: 1;
    }
    
    .property-label {
      font-size: 11px;
      font-weight: 600;
      fill: #2c3e50;
      dominant-baseline: middle;
    }
    
    .property-value {
      font-size: 11px;
      font-family: 'Courier New', monospace;
      fill: #1565c0;
      dominant-baseline: middle;
    }
    
    .connection-label {
      font-size: 11px;
      font-weight: bold;
      dominant-baseline: middle;
    }
    
    .true-label {
      fill: #27ae60;
    }
    
    .false-label {
      fill: #e74c3c;
    }
    
    .flowchart-node {
      cursor: pointer;
      transition: all 0.3s ease;
    }
    
    .flowchart-node:hover .node-bg {
      filter: drop-shadow(4px 4px 8px rgba(0,0,0,0.3));
      transform: scale(1.02);
    }
    
    .flowchart-connection {
      pointer-events: none;
    }
    
    .legend {
      position: fixed;
      top: 20px;
      right: 20px;
      background: white;
      padding: 15px;
      border-radius: 8px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.2);
      font-size: 12px;
    }
    
    .legend-item {
      display: flex;
      align-items: center;
      margin: 5px 0;
      gap: 10px;
    }
    
    .legend-color {
      width: 20px;
      height: 15px;
      border-radius: 3px;
    }
  </style>
</head>
<body>
  <div class="flowchart-container">
    <div class="flowchart-header">
      <h1 class="flowchart-title">🌳 UML Flowchart Decision Tree</h1>
      <div class="flowchart-stats">
        <span>📊 Tree: ${metadata.treeIndex}</span>
        <span>📏 Depth: ${metadata.maxDepth}</span>
        <span>🔢 Nodes: ${metadata.nodeCount}</span>
        <span>🍃 Leaves: ${metadata.leafCount}</span>
      </div>
    </div>
    
    <div class="flowchart-controls">
      <button class="flowchart-button" onclick="zoomIn()">🔍 Zoom In</button>
      <button class="flowchart-button" onclick="zoomOut()">🔍 Zoom Out</button>
      <button class="flowchart-button" onclick="resetZoom()">🔄 Reset Zoom</button>
      <button class="flowchart-button" onclick="highlightPath()">✨ Highlight Paths</button>
    </div>
    
    <svg class="flowchart-svg" width="${maxX}" height="${maxY}" id="flowchart-svg">
      <defs>
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
          <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#ddd" stroke-width="1"/>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid)"/>
      ${generateFlowchartConnections()}
      ${generateFlowchartNodes()}
    </svg>
  </div>
  
  <div class="legend">
    <h4 style="margin: 0 0 10px 0;">Legend</h4>
    <div class="legend-item">
      <div class="legend-color" style="background: #ebf3fd; border: 1px solid #3498db;"></div>
      <span>Decision Node</span>
    </div>
    <div class="legend-item">
      <div class="legend-color" style="background: #e8f5e8; border: 1px solid #27ae60;"></div>
      <span>Leaf Node</span>
    </div>
    <div class="legend-item">
      <div class="legend-color" style="background: #27ae60;"></div>
      <span>True Path (≤)</span>
    </div>
    <div class="legend-item">
      <div class="legend-color" style="background: #e74c3c;"></div>
      <span>False Path (>)</span>
    </div>
  </div>

  <script>
    let currentZoom = 1;
    const svg = document.getElementById('flowchart-svg');
    
    function zoomIn() {
      currentZoom *= 1.2;
      applyzoom();
    }
    
    function zoomOut() {
      currentZoom /= 1.2;
      applyzoom();
    }
    
    function resetZoom() {
      currentZoom = 1;
      applyzoom();
    }
    
    function applyzoom() {
      svg.style.transform = \`scale(\${currentZoom})\`;
      svg.style.transformOrigin = 'top left';
    }
    
    function highlightPath() {
      const connections = document.querySelectorAll('.flowchart-connection');
      connections.forEach((conn, index) => {
        setTimeout(() => {
          conn.style.opacity = '0.3';
          setTimeout(() => {
            conn.style.opacity = '1';
          }, 200);
        }, index * 100);
      });
    }
    
    // Add click handlers to nodes
    document.querySelectorAll('.flowchart-node').forEach(node => {
      node.addEventListener('click', function() {
        const nodeId = this.id.replace('node-', '');
        alert(\`Node \${nodeId} clicked!\\nYou can add custom interactions here.\`);
      });
    });
    
    // Add pan functionality
    let isPanning = false;
    let startX, startY, currentTranslateX = 0, currentTranslateY = 0;
    
    svg.addEventListener('mousedown', (e) => {
      isPanning = true;
      startX = e.clientX - currentTranslateX;
      startY = e.clientY - currentTranslateY;
      svg.style.cursor = 'grabbing';
    });
    
    document.addEventListener('mousemove', (e) => {
      if (!isPanning) return;
      
      currentTranslateX = e.clientX - startX;
      currentTranslateY = e.clientY - startY;
      
      svg.style.transform = \`scale(\${currentZoom}) translate(\${currentTranslateX}px, \${currentTranslateY}px)\`;
    });
    
    document.addEventListener('mouseup', () => {
      isPanning = false;
      svg.style.cursor = 'grab';
    });
    
    svg.style.cursor = 'grab';
  </script>
</body>
</html>`;
  }
} 